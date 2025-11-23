import cv2
import math
from datetime import datetime

import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase,
    RTCConfiguration,
)
import av

# ----------------- 전역 설정 -----------------
st.set_page_config(page_title="기준 사진 각도 맞추기", layout="centered")

st.title("📸 기준 사진 각도 맞추기 자동 촬영")
st.write(
    "1) 기준이 될 얼굴 사진을 업로드하고\n"
    "2) 웹캠에서 그 사진과 비슷한 각도로 얼굴을 맞추면\n"
    "3) 자동으로 캡쳐되고, 아래에서 다운로드할 수 있어요."
)

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

mp_face_mesh = mp.solutions.face_mesh

# 기준 사진에서 계산한 특징값 (roll, horiz, vert)
TARGET_FEATURES = None

# 허용 오차 기본값
TOL_ROLL_DEG = 8.0     # 얼굴 기울기(롤) 허용 오차 (도)
TOL_POS = 0.15         # 코 위치(좌우/상하) 허용 오차 (정규화 단위)


# ----------------- 얼굴 특징 계산 함수 -----------------
def compute_face_features(landmarks):
    """
    FaceMesh 랜드마크(468개)에서
    - 두 눈의 기울기(roll 비슷)
    - 코의 좌우/상하 위치(눈 기준, yaw/pitch 비슷)
    3개 특징값 반환
    """
    # 주요 포인트 인덱스
    idx_nose = 1
    idx_left_eye = 33
    idx_right_eye = 263
    idx_chin = 152

    nose = landmarks[idx_nose]
    le = landmarks[idx_left_eye]
    re = landmarks[idx_right_eye]
    chin = landmarks[idx_chin]

    # 좌표 (정규화된 0~1)
    nose_xy = (nose.x, nose.y)
    le_xy = (le.x, le.y)
    re_xy = (re.x, re.y)
    chin_xy = (chin.x, chin.y)

    # 두 눈 중심
    eyes_center = ((le_xy[0] + re_xy[0]) / 2, (le_xy[1] + re_xy[1]) / 2)

    # 눈 사이 벡터
    eye_dx = re_xy[0] - le_xy[0]
    eye_dy = re_xy[1] - le_xy[1]
    eye_dist = math.sqrt(eye_dx**2 + eye_dy**2) + 1e-6

    # 1) Roll: 두 눈의 기울기 (반시계 방향 +)
    roll_rad = math.atan2(eye_dy, eye_dx)
    roll_deg = math.degrees(roll_rad)

    # 2) 수평 위치: 코가 눈 중간에서 좌우로 얼마나 치우쳤는지
    horiz = (nose_xy[0] - eyes_center[0]) / eye_dist

    # 3) 수직 위치: 코가 눈 중간에서 위/아래로 얼마나 치우쳤는지
    vert = (nose_xy[1] - eyes_center[1]) / eye_dist

    return roll_deg, horiz, vert


# ----------------- 기준 사진 업로드 & 분석 -----------------
st.subheader("① 기준 사진 업로드")

uploaded = st.file_uploader("기준이 될 얼굴 사진을 업로드하세요", type=["jpg", "jpeg", "png"])

if "target_features" not in st.session_state:
    st.session_state["target_features"] = None

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="기준 사진", use_column_width=True)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        target_roll, target_horiz, target_vert = compute_face_features(lm)
        st.session_state["target_features"] = (target_roll, target_horiz, target_vert)

        st.success(
            f"기준 얼굴 각도 분석 완료!\n\n"
            f"- Roll: {target_roll:.1f}°\n"
            f"- Horiz: {target_horiz:.2f}\n"
            f"- Vert: {target_vert:.2f}"
        )
    else:
        st.error("얼굴을 찾지 못했어요. 얼굴이 잘 보이도록 된 사진을 다시 업로드해 주세요.")

# 전역 변수에 반영
TARGET_FEATURES = st.session_state["target_features"]

st.markdown("---")

# 허용 오차 슬라이더
st.subheader("② 각도 허용 범위 설정")

TOL_ROLL_DEG = st.slider("얼굴 기울기 허용 오차 (Roll, 도)", 2.0, 20.0, 8.0, 0.5)
TOL_POS = st.slider("코 위치 허용 오차 (좌우/상하, 값이 작을수록 까다로움)", 0.05, 0.5, 0.15, 0.01)

st.markdown("---")
st.subheader("③ 웹캠으로 기준 각도 맞추기")


# ----------------- 비디오 프로세서 -----------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.last_capture = None
        self.match_count = 0
        self.cooldown = 0  # 너무 자주 캡쳐되는 것 방지용

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        # 기준 각도가 없으면 안내만 표시
        if TARGET_FEATURES is None:
            cv2.putText(
                img,
                "Please upload target photo first.",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        target_roll, target_h, target_v = TARGET_FEATURES

        matched_now = False

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # 얼굴 폴리라인 대충 보여주기 (눈, 코, 턱 정도만 점 찍기)
            for idx in [1, 33, 263, 152]:
                p = lm[idx]
                cx, cy = int(p.x * w), int(p.y * h)
                cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)

            roll, horiz, vert = compute_face_features(lm)

            # 차이 계산
            d_roll = abs(roll - target_roll)
            d_h = abs(horiz - target_h)
            d_v = abs(vert - target_v)

            text = f"diff R:{d_roll:.1f} H:{d_h:.2f} V:{d_v:.2f}"
            cv2.putText(
                img,
                text,
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 쿨다운 감소
            if self.cooldown > 0:
                self.cooldown -= 1

            # 허용 범위 안에 들어왔는지 체크
            if (
                d_roll <= TOL_ROLL_DEG
                and d_h <= TOL_POS
                and d_v <= TOL_POS
                and self.cooldown == 0
            ):
                matched_now = True
                self.match_count += 1
                self.cooldown = 40  # 대략 40프레임 동안은 재캡쳐 X

                self.last_capture = img.copy()

                cv2.putText(
                    img,
                    "MATCHED! CAPTURED!",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 0, 255),
                    4,
                    cv2.LINE_AA,
                )

        # 좌측 상단에 매칭 횟수 표시
        cv2.putText(
            img,
            f"Match count: {self.match_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ----------------- WebRTC 스트림 -----------------
ctx = webrtc_streamer(
    key="face-angle-match",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.caption("📌 웹캠이 안 보이면: 다른 프로그램(카톡, 줌, 브라우저 탭)이 카메라를 잡고 있지 않은지 확인해 주세요.")

st.markdown("---")
st.subheader("④ 캡쳐된 사진 불러오기 & 다운로드")

refresh = st.button("🔄 최신 캡쳐 불러오기")
placeholder = st.empty()

if not ctx.video_processor:
    placeholder.info("위에서 START 버튼을 눌러 웹캠을 먼저 켜주세요.")
else:
    vp = ctx.video_processor

    if refresh:
        if vp.last_capture is None:
            placeholder.warning("아직 캡쳐된 사진이 없어요. 기준 각도를 맞춰보세요!")
        else:
            img_rgb = cv2.cvtColor(vp.last_capture, cv2.COLOR_BGR2RGB)
            placeholder.image(
                img_rgb,
                caption=f"Matched! (총 {vp.match_count}회)",
                use_column_width=True,
            )

            _, buf = cv2.imencode(".png", vp.last_capture)
            st.download_button(
                "📥 이 사진 다운로드",
                data=buf.tobytes(),
                file_name=f"matched_face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
            )
    else:
        placeholder.info("기준 사진 각도를 맞춘 뒤, 위의 🔄 버튼을 눌러 캡쳐된 사진을 확인하세요.")
