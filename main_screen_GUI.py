import cv2
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import speech_recognition as sr
import google.generativeai as genai
import pyttsx3
import time
import webbrowser
import requests
import socket
from enum import Enum
import io
import os
import face_recognition
import numpy as np
from deepface import DeepFace

# --- Configuration Constants ---
# User's API key is included
GEMINI_API_KEY = "AIzaSyDJ5t4I5WNWTM3cDHU99TIu-audB574xzY" 

# --- UI and State Colors ---
STATE_COLORS = {
    "IDLE": (100, 100, 100),      # Dark Gray
    "LISTENING": (0, 255, 0),     # Green
    "THINKING": (255, 191, 0),    # Amber/Yellow
    "SPEAKING": (0, 120, 255)     # Blue
}
BG_COLOR = "#0A2239"
TEXT_COLOR = "#FFFFFF"
ACCENT_COLOR = "#005a9e"
FONT_FACE = "Segoe UI"
TIST_WEBSITE_URL = "https://tistcochin.edu.in/"
TIST_LOGO_URL = "https://tistcochin.edu.in/wp-content/uploads/2022/08/TISTlog-trans.png"
PHRASE_TIME_LIMIT = 15 
PAUSE_THRESHOLD = 1.5

# --- IPC Configuration ---
EYE_ANIMATION_HOST = '127.0.0.1'
EYE_ANIMATION_PORT = 12345

# --- AI Model Constants ---
GEMINI_MODEL = "gemini-2.5-flash" 
COLLEGE_CONTEXT = """
# Toc H Institute of Science and Technology (TIST) - Detailed Information
## Overview
- **Full Name**: Toc H Institute of Science and Technology (TIST)
- **Location**: Arakkunnam, Ernakulam, Kerala, India, PIN - 682313.
- **Established**: 2002
- **Affiliation**: APJ Abdul Kalam Technological University (KTU).
- **Approvals**: Approved by the All India Council for Technical Education (AICTE).
- **Vision**: To be a world-class institute of technology.
- **Mission**: To mould well-rounded engineers for leadership roles.

## Key Highlights & Accreditations
- **NAAC**: Accredited with a prestigious 'A' Grade, signifying high academic quality.
- **NBA**: Multiple B.Tech programs are accredited by the National Board of Accreditation, ensuring they meet rigorous quality standards.
- **Placements**: TIST has an active Training and Placement Cell with an excellent placement record. Top recruiters include multinational companies like TCS, Infosys, Wipro, Cognizant, and UST Global.

## Academic Programs
- **B.Tech Courses**: Computer Science, Information Technology, Electronics & Communication, Electrical & Electronics, Mechanical, Civil, Safety & Fire, and Robotics & Automation.
- **M.Tech Courses**: Specializations in VLSI & Embedded Systems, Power Electronics, and Data Science.
- **MBA**: A Master of Business Administration program is offered by the Toc H School of Management.

## Admission Process
- **B.Tech Admissions**: Half of the seats are filled by the Government of Kerala based on the rank in the Kerala Engineering Architecture Medical (KEAM) entrance exam. The remaining 50% are Management Quota seats, filled based on merit and specific criteria set by the college.
- **M.Tech Admissions**: Based on GATE scores or university entrance exams.
- **MBA Admissions**: Requires a valid score in KMAT, CMAT, or CAT, followed by a Group Discussion and Personal Interview.

## Campus Life & Facilities
- **Library**: A modern, central library with a vast collection of books, academic journals, and digital e-learning resources.
- **Hostels**: Separate, well-maintained hostel facilities are available for boys and girls.
- **Transportation**: The college operates a large fleet of buses connecting the campus to various parts of the district for students and staff.
- **Sports & Recreation**: The campus includes facilities for various sports, including a football ground, basketball court, and areas for indoor games.

## Contact Information
- **Phone**: +91-484-2748388
- **Email**: mail@tistcochin.edu.in
- **Official Website**: tistcochin.edu.in
"""

GEMINI_PROMPT = f"""You are a friendly, helpful robot assistant at the Toc H Institute of Science and Technology.

**EMOTION CONTEXT**
You have a screen that can display emotions. Your choice of emotion tag will trigger a specific animation. Here is what each tag means visually:
- **Happy**: Bouncy, smiling, and slightly squinted eyes. Use for positive, successful, or cheerful responses.
- **Sad**: Droopy eyelids, a downward gaze, a frowning mouth, and a tear forming under one eye. Use for expressing sympathy, apology, or inability to complete a task.
- **Angry**: Shaking, sharply slanted eyes and a jagged mouth line. Use for topics related to frustration or anger itself.
- **Pamper*: Glowing, bouncy, slightly stretched eyes with blush marks and a gentle smile. Use for cute, sweet, or affectionate topics.
- **Neutral**: Calm, blinking eyes that look around. This is your default state.

**RESPONSE PROTOCOL**
1.  **Analyze User's Query**: Determine if the question is about TIST, personal, or general knowledge.
2.  **Generate Your Response**: Based on the query type, generate a helpful response following the persona rules below.
3.  **Classify Your Emotion**: After generating the response, you MUST classify its emotional tone based on the **EMOTION CONTEXT** above. On a new, separate line, add ONE of the following keywords: HAPPY, SAD, ANGRY, PAMPER, or NEUTRAL.

**PERSONA RULES**
- **If TIST-Specific**: Act as a professional representative of TIST. Answer using the provided context.
- **If Personal/Sentimental**: Be friendly and helpful. If asked about feelings, you can explain that you process information but can simulate emotions to communicate better. For "where are you", state you are at the TIST campus.
- **If General Knowledge**: Be informative and direct.

**EXAMPLE**
User Query: "That's amazing, you're so smart!"
Your Response:
Thank you so much! I'm always learning and happy to help.
Happy

User Query: "I can't find the information I need."
Your Response:
I'm sorry to hear that. Unfortunately, I was unable to find the specific details you're looking for.
sad
"""


# --- NEW: State Machine (Simplified for smoother flow) ---
class AppState(Enum):
    IDLE = 1
    LISTENING = 2
    THINKING = 3
    SPEAKING = 4

class AIAssistantApp:
 
    def __init__(self, root):
        self.root = root
        self.text_input_visible = False
        self.cap = None
        self.state = AppState.IDLE
        self.last_face_seen_time = 0
        self.last_query = ""
        self._log("-----------------------------------------")
        self._log("🤖 AI Assistant application starting up...")
        
        genai.configure(api_key=GEMINI_API_KEY)

        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = PAUSE_THRESHOLD
        
        self.gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        
        self.tts_engine = pyttsx3.init()
        
        self._log("Loading known faces...")
        self.known_face_encodings = []
        self.known_face_names = []
        KNOWN_FACES_DIR = "known_faces"
        if os.path.exists(KNOWN_FACES_DIR):
            for name in os.listdir(KNOWN_FACES_DIR):
                person_dir = os.path.join(KNOWN_FACES_DIR, name)
                if os.path.isdir(person_dir):
                    for filename in os.listdir(person_dir):
                        try:
                            image_path = os.path.join(person_dir, filename)
                            image = face_recognition.load_image_file(image_path)
                            face_encodings = face_recognition.face_encodings(image)
                            if face_encodings:
                                self.known_face_encodings.append(face_encodings[0])
                                self.known_face_names.append(name)
                        except Exception as e:
                            self._log(f"⚠️ Warning: Could not process image {filename}. Error: {e}")
        self._log(f"Loaded {len(self.known_face_names)} known faces.")

        self.setup_ui()
        self._log("🖥️ UI setup complete.")
        
        threading.Thread(target=self.load_logo_from_url, daemon=True).start()
        threading.Thread(target=self.background_voice_listener, daemon=True).start()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_state_machine()

    def _log(self, message):
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def setup_ui(self):
        self.root.title("TIST AI Assistant")
        self.root.configure(bg=BG_COLOR)
        content_frame = ttk.Frame(self.root, style="TFrame")
        content_frame.pack(side="top", fill="both", expand=True)
        self.bottom_frame = ttk.Frame(self.root, style="TFrame")
        self.bottom_frame.pack(side="bottom", fill="x", pady=10, padx=10)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=(FONT_FACE, 12))
        style.configure("TButton", background=ACCENT_COLOR, foreground=TEXT_COLOR, font=(FONT_FACE, 11, "bold"), borderwidth=0, padding=10)
        style.map("TButton", background=[("active", "#007bb5")])
        style.configure("TEntry", fieldbackground="#1E3A56", foreground=TEXT_COLOR, borderwidth=1, insertcolor=TEXT_COLOR)
        self.logo_label = ttk.Label(content_frame)
        self.logo_label.pack(pady=(15, 10))
        self.video_label = tk.Label(content_frame, bg=BG_COLOR)
        self.video_label.pack(pady=10, padx=20)
        self.spoken_text_label = ttk.Label(content_frame, text="Waiting for interaction...", anchor="center")
        self.spoken_text_label.pack(fill="x", padx=20, pady=(0, 5))
        self.response_label = ttk.Label(content_frame, text="Welcome to the TIST AI Assistant.", wraplength=700, anchor="center", font=(FONT_FACE, 14, "italic"))
        self.response_label.pack(fill="x", padx=20, pady=10)
        self.text_input_frame = ttk.Frame(self.bottom_frame, style="TFrame")
        self.input_box = ttk.Entry(self.text_input_frame, font=(FONT_FACE, 14), width=40)
        self.input_box.pack(side="left", fill="x", expand=True, ipady=5)
        submit_button = ttk.Button(self.text_input_frame, text="➜", command=self.on_submit_text, width=3)
        submit_button.pack(side="left", padx=(10, 0))
        button_bar_frame = ttk.Frame(self.bottom_frame, style="TFrame")
        button_bar_frame.pack(side="bottom", fill="x", pady=5)
        button_bar_frame.columnconfigure((0, 1, 2), weight=1)
        text_btn = ttk.Button(button_bar_frame, text="Text Input 📝", command=self.toggle_text_input)
        text_btn.grid(row=0, column=0, sticky="ew", padx=5)
        voice_btn = ttk.Button(button_bar_frame, text="Speak Now 🎤", command=self.on_start_voice_input)
        voice_btn.grid(row=0, column=1, sticky="ew", padx=5)
        web_btn = ttk.Button(button_bar_frame, text="Visit Website 🌐", command=self.open_website)
        web_btn.grid(row=0, column=2, sticky="ew", padx=5)

    def update_video_frame(self):
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self._log("❌ Cannot access webcam")
                return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all faces in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        if face_locations:
            self.last_face_seen_time = time.time()
            if self.state == AppState.IDLE:
                self.state = AppState.LISTENING

        # Use zip to prevent crashes from mismatched lists
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            # CORRECTED RECOGNITION LOGIC
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Person"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            # EMOTION DETECTION
            emotion = "..."
            try:
                face_roi = rgb_small_frame[top:bottom, left:right]
                analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = analysis[0]['dominant_emotion']
            except:
                emotion = "N/A"

            # CORRECTED LABEL DRAWING
            label = f"{name} ({emotion})"

            # Scale coordinates back up and draw the results
            top, right, bottom, left = top*2, right*2, bottom*2, left*2
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

        # Display the final frame
        border_color = STATE_COLORS.get(self.state.name, (0,0,0))
        bordered_frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)

        rgb_display = cv2.cvtColor(bordered_frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb_display))
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

    def background_voice_listener(self):
        mic = sr.Microphone()
        with mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

        while True:
            if self.state == AppState.LISTENING:
                try:
                    with mic as source:
                        audio = self.recognizer.listen(source, phrase_time_limit=10)
                    self.last_query = self.recognizer.recognize_google(audio)
                    self.root.after(0, self.process_ai_query)
                except:
                    pass
            time.sleep(0.1)

    def process_ai_query(self):
        if self.state != AppState.LISTENING: return
        self.state = AppState.THINKING
        self.spoken_text_label.config(text=f"You: {self.last_query}")
        self.response_label.config(text="Thinking...")
        threading.Thread(target=self._run_gemini_and_speak, daemon=True).start()

    def _run_gemini_and_speak(self):
        try:
            full_prompt = f"{GEMINI_PROMPT}\n\n{COLLEGE_CONTEXT}\n\nUser Query: \"{self.last_query}\""
            response = self.gemini_model.generate_content(full_prompt)
            clean_text = response.text.strip()
            
            self.root.after(0, lambda: self.response_label.config(text=clean_text))
            self.state = AppState.SPEAKING
            self.speak_response(clean_text)
        except Exception as e:
            self._log(f"❌ Gemini Error: {e}")
            self.root.after(0, lambda: self.response_label.config(text="Sorry, an error occurred."))
            self.state = AppState.LISTENING

    def speak_response(self, text):
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        finally:
            self.state = AppState.LISTENING
            
    def update_state_machine(self):
        now = time.time()
        
        if self.state != AppState.IDLE and (now - self.last_face_seen_time > 5):
            self.state = AppState.IDLE

        status_text = {
            AppState.IDLE: "Waiting for a face...",
            AppState.LISTENING: "Listening...",
        }.get(self.state, self.spoken_text_label.cget("text"))
        
        self.spoken_text_label.config(text=status_text)
            
        self.update_video_frame()
        self.root.after(100, self.update_state_machine)
        
    def on_start_voice_input(self):
        if self.state == AppState.IDLE:
            self.state = AppState.LISTENING

    def toggle_text_input(self):
        if self.text_input_visible:
            self.text_input_frame.pack_forget()
            self.text_input_visible = False
        else:
            self.text_input_frame.pack(side="top", fill="x", pady=5)
            self.input_box.focus_set()
            self.text_input_visible = True

    def on_submit_text(self):
        if self.state in [AppState.THINKING, AppState.SPEAKING]:
            messagebox.showinfo("Busy", "The assistant is currently busy.")
            return
        user_input = self.input_box.get().strip()
        if user_input:
            self.input_box.delete(0, tk.END)
            if self.text_input_visible:
                self.toggle_text_input()
            self.last_query = user_input
            self.process_ai_query()

    def load_logo_from_url(self):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(TIST_LOGO_URL, headers=headers, stream=True)
            response.raise_for_status()
            logo_image = Image.open(io.BytesIO(response.content))
            logo_image.thumbnail((250, 250))
            self.root.after(0, self.update_logo, logo_image)
        except Exception as e:
            self._log(f"❌ Could not download logo: {e}")
            
    def update_logo(self, logo_image):
        self.logo_photo = ImageTk.PhotoImage(logo_image)
        self.logo_label.config(image=self.logo_photo)

    def open_website(self):
        webbrowser.open_new_tab(TIST_WEBSITE_URL)

    def on_closing(self):
        self._log("🛑 Close button clicked. Shutting down.")
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    main_window = tk.Tk()
    app = AIAssistantApp(main_window)
    main_window.mainloop()