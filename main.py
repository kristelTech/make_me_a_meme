import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pickle
import os


class MemeMatcher:
    # MediaPipe landmark indices
    LEFT_EYE_UPPER = [159, 145, 158]
    LEFT_EYE_LOWER = [23, 27, 133]
    RIGHT_EYE_UPPER = [386, 374, 385]
    RIGHT_EYE_LOWER = [253, 257, 362]
    LEFT_EYEBROW = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [300, 293, 334, 296, 336]
    MOUTH_OUTER = [61, 291, 39, 181, 0, 17, 269, 405]
    MOUTH_INNER = [78, 308, 95, 88]
    NOSE_TIP = 4

    CACHE_FILE = "meme_features_cache.pkl"

    def __init__(self, assets_folder="assets", frame_skip=2, meme_height=480):
        self.last_features = None
        self.frame_counter = 0
        self.frame_skip = frame_skip
        self.meme_height = meme_height

        # Download or load models
        self.face_model_path = self._download_model(
            "face_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        self.hand_model_path = self._download_model(
            "hand_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

        # MediaPipe landmarkers
        self.face_mesh_video = self._init_face_landmarker(video_mode=True)
        self.hand_detector_video = self._init_hand_landmarker(video_mode=True)
        self.face_mesh_image = self._init_face_landmarker(video_mode=False)
        self.hand_detector_image = self._init_hand_landmarker(video_mode=False)

        # Meme storage
        self.memes = []
        self.meme_features = []

        # Feature vectors for similarity computation
        self.feature_keys = [
            'surprise_score', 'smile_score', 'concern_score', 'cheers_score',
            'hand_raised', 'num_hands', 'eye_openness', 'eyes_symmetry',
            'mouth_openness', 'mouth_width_ratio', 'mouth_elevation',
            'eyebrow_height', 'brow_symmetry'
        ]
        self.feature_weights = np.array([25, 20, 20, 30, 25, 15, 20, 10, 25, 20, 15, 20, 10])
        self.feature_factors = np.array([10, 10, 10, 10, 15, 15, 5, 5, 5, 5, 5, 5, 5])

        # Load memes (with caching)
        self.load_memes(assets_folder)

    # ---------------- Model Initialization ----------------
    def _download_model(self, model_path, url):
        import subprocess
        if not os.path.exists(model_path):
            print(f"Downloading {model_path}...")
            try:
                subprocess.run(['curl', '-L', url, '-o', model_path], check=True, capture_output=True)
                print(f"{model_path} downloaded successfully!")
            except subprocess.CalledProcessError:
                raise RuntimeError(f"Failed to download model. Please download manually from {url}")
        return model_path

    def _init_face_landmarker(self, video_mode=True):
        mode = mp.tasks.vision.RunningMode.VIDEO if video_mode else mp.tasks.vision.RunningMode.IMAGE
        return mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.face_model_path),
                running_mode=mode,
                num_faces=1,
                min_face_detection_confidence=0.5 if video_mode else 0.3,
                min_face_presence_confidence=0.5 if video_mode else 0.3,
                min_tracking_confidence=0.5 if video_mode else 0.0
            )
        )

    def _init_hand_landmarker(self, video_mode=True):
        mode = mp.tasks.vision.RunningMode.VIDEO if video_mode else mp.tasks.vision.RunningMode.IMAGE
        return mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.hand_model_path),
                running_mode=mode,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3 if video_mode else 0.0
            )
        )

    # ---------------- Meme Loading with Cache ----------------
    def load_memes(self, folder):
        cache_exists = os.path.exists(self.CACHE_FILE)
        if cache_exists:
            with open(self.CACHE_FILE, "rb") as f:
                self.memes, self.meme_features = pickle.load(f)
            print(f"Loaded {len(self.memes)} memes from cache.\n")
            return

        assets_path = Path(folder)
        image_files = list(assets_path.glob("*.jpg")) + list(assets_path.glob("*.png")) + list(assets_path.glob("*.jpeg"))
        print(f"Found {len(image_files)} meme images. Extracting features...")

        def process_meme(img_file):
            img = cv2.imread(str(img_file))
            if img is None:
                return None
            h, w = img.shape[:2]
            scale = self.meme_height / h
            img_resized = cv2.resize(img, (int(w * scale), self.meme_height))
            features = self.extract_face_features(img_resized, is_static=True)
            if features is None:
                print(f"Skipping {img_file.name}: No face detected")
                return None
            return {'image': img_resized, 'name': img_file.stem.replace('_', ' ').title(),
                    'path': str(img_file)}, features

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_meme, sorted(image_files)))

        for r in results:
            if r:
                meme, features = r
                self.memes.append(meme)
                self.meme_features.append(features)
                print(f"Loaded: {meme['name']}")

        with open(self.CACHE_FILE, "wb") as f:
            pickle.dump((self.memes, self.meme_features), f)
        print(f"Total memes loaded: {len(self.memes)}\n")

    # ---------------- Feature Extraction ----------------
    def extract_face_features(self, image, is_static=False):
        if is_static:
            face_landmarker = self.face_mesh_image
            hand_landmarker = self.hand_detector_image
        else:
            face_landmarker = self.face_mesh_video
            hand_landmarker = self.hand_detector_video

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if is_static:
            face_res = face_landmarker.detect(mp_image)
            hand_res = hand_landmarker.detect(mp_image)
        else:
            self.frame_counter += 1
            if self.frame_counter % self.frame_skip != 0:
                return getattr(self, "last_features", None)
            face_res = face_landmarker.detect_for_video(mp_image, self.frame_counter)
            hand_res = hand_landmarker.detect_for_video(mp_image, self.frame_counter)

        if not face_res.face_landmarks:
            return None

        landmarks = face_res.face_landmarks[0]
        landmark_array = np.array([[l.x, l.y] for l in landmarks])
        features = self._compute_features(landmark_array, hand_res)
        self.last_features = features
        return features

# Live with DataCamp
    def _compute_features(self, landmark_array, hand_res):
        #TODO Eye aspect ratios with DataCamp


        # TODO Mouth with DataCamp



        # #eyebrows
        # left_brow_y = landmark_array[self.LEFT_EYEBROW][:, 1].mean()
        # right_brow_y = landmark_array[self.RIGHT_EYEBROW][:, 1].mean()
        # left_eye_center = landmark_array[self.LEFT_EYE_UPPER + self.LEFT_EYE_LOWER][:, 1].mean()
        # right_eye_center = landmark_array[self.RIGHT_EYE_UPPER + self.RIGHT_EYE_LOWER][:, 1].mean()
        # left_brow_h = left_eye_center - left_brow_y
        # right_brow_h = right_eye_center - right_brow_y
        # avg_brow_h = (left_brow_h + right_brow_h) / 2.0
        #
        # mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2.0
        # nose_tip = landmark_array[self.NOSE_TIP]
        # mouth_elev = nose_tip[1] - mouth_center_y

        #TODO Hands with DataCamp

        # return {
        #     'eye_openness': avg_ear,
        #     'left_eye_open': left_ear,
        #     'right_eye_open': right_ear,
        #     'eyes_symmetry': abs(left_ear - right_ear),
        #     'mouth_openness': mouth_ar,
        #     'mouth_width': mouth_width,
        #     'mouth_width_ratio': mouth_width_ratio,
        #     'mouth_elevation': mouth_elev,
        #     'eyebrow_height': avg_brow_h,
        #     'left_brow_height': left_brow_h,
        #     'right_brow_height': right_brow_h,
        #     'brow_symmetry': abs(left_brow_h - right_brow_h),
        #     'num_hands': num_hands,
        #     'hand_raised': hand_raised,
        #     'surprise_score': avg_ear * avg_brow_h * mouth_ar,
        #     'smile_score': mouth_width_ratio * (1.0 - mouth_ar),
        #     'concern_score': avg_brow_h * (1.0 - mouth_elev),
        #     'cheers_score': mouth_width_ratio * (1.0 - mouth_ar) * hand_raised
        # }

                pass



    # ---------------- Vectorized Similarity ----------------

#TODO compute_similarity with DataCamp
    def compute_similarity(self):
        pass

# TODO find_best_match with DataCamp
    def find_best_match(self,user_features):
        pass

    # ---------------- Main Loop ----------------
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("\n🎥 Camera started! Press 'q' to quit\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            user_features = self.extract_face_features(frame)
            best_meme, score = self.find_best_match(user_features)

            h, w = frame.shape[:2]
            if best_meme:
                meme_img = best_meme['image']
                meme_h, meme_w = meme_img.shape[:2]
                scale = h / meme_h
                new_w = int(meme_w * scale)
                meme_resized = cv2.resize(meme_img, (new_w, h))

                display = np.zeros((h, w + new_w, 3), dtype=np.uint8)
                display[:, :w] = frame
                display[:, w:w + new_w] = meme_resized

                # Overlay text
                cv2.rectangle(display, (5, 5), (200, 45), (0, 0, 0), -1)
                cv2.putText(display, "YOU", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.rectangle(display, (w + 5, 5), (w + new_w - 5, 75), (0, 0, 0), -1)
                cv2.putText(display, best_meme['name'], (w + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, f"Match: {score:.1f}", (w + 10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                display = frame
                cv2.putText(display, "No face detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Meme Matcher - Press Q to quit", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Meme Matcher Starting...\n")
    matcher = MemeMatcher(assets_folder="assets")
    matcher.run()