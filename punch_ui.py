import customtkinter as ctk
import time
import threading
import winsound
from read_arduino import run_reader

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class PunchDetectorUI:
    def __init__(self):
        self.app = ctk.CTk()
        self.app.geometry("620x680")
        self.app.title("Punch Detector")

        self.session_running = False
        self.session_paused = False
        self.start_time = 0
        self.elapsed_before_pause = 0
        self.latest_punch = "WAITING..."
        self.recent_punches = []
        self.last_ko_time = 0
        self.pending_ko_sound = False

        self.build_ui()
        self.refresh_buttons()

        threading.Thread(
            target=run_reader,
            args=(self.handle_punch,),
            daemon=True
        ).start()

        self.update_timer()
        self.update_punch_label()
        self.update_history()
        self.update_sound()

    def build_ui(self):
        self.app.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.app,
            text="Punch Detector",
            font=("Arial", 34, "bold")
        )
        self.title_label.pack(pady=(24, 8))

        self.status_label = ctk.CTkLabel(
            self.app,
            text="Session Idle",
            font=("Arial", 16),
            text_color="gray70"
        )
        self.status_label.pack(pady=(0, 18))

        self.main_card = ctk.CTkFrame(self.app, corner_radius=18)
        self.main_card.pack(padx=24, pady=10, fill="x")

        self.punch_title = ctk.CTkLabel(
            self.main_card,
            text="Latest Punch",
            font=("Arial", 18, "bold")
        )
        self.punch_title.pack(pady=(18, 6))

        self.punch_label = ctk.CTkLabel(
            self.main_card,
            text="WAITING...",
            font=("Arial", 42, "bold")
        )
        self.punch_label.pack(pady=(0, 18))

        self.timer_label = ctk.CTkLabel(
            self.main_card,
            text="Session Time: 00:00",
            font=("Arial", 22)
        )
        self.timer_label.pack(pady=(0, 18))

        self.history_card = ctk.CTkFrame(self.app, corner_radius=18)
        self.history_card.pack(padx=24, pady=10, fill="x")

        self.history_title = ctk.CTkLabel(
            self.history_card,
            text="Recent Punches",
            font=("Arial", 18, "bold")
        )
        self.history_title.pack(pady=(16, 10))

        self.history_label = ctk.CTkLabel(
            self.history_card,
            text="No punches yet",
            font=("Arial", 18),
            justify="left",
            anchor="w"
        )
        self.history_label.pack(padx=18, pady=(0, 16), anchor="w")

        self.button_frame = ctk.CTkFrame(self.app, fg_color="transparent")
        self.button_frame.pack(pady=24)

        self.start_button = ctk.CTkButton(
            self.button_frame,
            text="Start Session",
            font=("Arial", 18, "bold"),
            width=170,
            height=48,
            corner_radius=12,
            command=self.start_session
        )

        self.pause_button = ctk.CTkButton(
            self.button_frame,
            text="Pause",
            font=("Arial", 18, "bold"),
            width=130,
            height=48,
            corner_radius=12,
            command=self.pause_session
        )

        self.stop_button = ctk.CTkButton(
            self.button_frame,
            text="Stop",
            font=("Arial", 18, "bold"),
            width=130,
            height=48,
            corner_radius=12,
            command=self.stop_session
        )

    def play_ko_sound(self):
        winsound.PlaySound(
             r"C:\Users\vladb\OneDrive\Documents\punchproject\ko_fixed.wav",
            winsound.SND_FILENAME | winsound.SND_ASYNC
        )

    def format_punch_display(self, punch, power):
        punch_text = str(punch).replace("_", " ").upper()

        try:
            power_int = int(power)
            return f"{punch_text}   {power_int}/10"
        except Exception:
            return punch_text

    def handle_punch(self, punch, power):
        display = self.format_punch_display(punch, power)
        self.latest_punch = display

        try:
            power_int = int(power)
        except Exception:
            power_int = -1

        if power_int == 10:
            now = time.time()
            if now - self.last_ko_time > 1.0:
                self.last_ko_time = now
                self.pending_ko_sound = True

        if self.session_running and not self.session_paused:
            self.recent_punches.insert(0, display)
            self.recent_punches = self.recent_punches[:5]

    def refresh_buttons(self):
        for widget in self.button_frame.winfo_children():
            widget.pack_forget()

        if not self.session_running:
            self.start_button.pack()
            self.status_label.configure(text="Session Idle")
        else:
            self.pause_button.pack(side="left", padx=10)
            self.stop_button.pack(side="left", padx=10)

            if self.session_paused:
                self.pause_button.configure(text="Resume")
                self.status_label.configure(text="Session Paused")
            else:
                self.pause_button.configure(text="Pause")
                self.status_label.configure(text="Session Active")

    def start_session(self):
        self.session_running = True
        self.session_paused = False
        self.elapsed_before_pause = 0
        self.start_time = time.time()
        self.latest_punch = "WAITING..."
        self.recent_punches.clear()

        self.punch_label.configure(text="WAITING...")
        self.history_label.configure(text="No punches yet")
        self.refresh_buttons()

    def pause_session(self):
        if not self.session_running:
            return

        if not self.session_paused:
            self.elapsed_before_pause = time.time() - self.start_time
            self.session_paused = True
        else:
            self.start_time = time.time() - self.elapsed_before_pause
            self.session_paused = False

        self.refresh_buttons()

    def stop_session(self):
        self.session_running = False
        self.session_paused = False
        self.start_time = 0
        self.elapsed_before_pause = 0
        self.latest_punch = "WAITING..."
        self.recent_punches.clear()

        self.timer_label.configure(text="Session Time: 00:00")
        self.punch_label.configure(text="WAITING...")
        self.history_label.configure(text="No punches yet")
        self.refresh_buttons()

    def update_timer(self):
        if self.session_running and not self.session_paused:
            elapsed = int(time.time() - self.start_time)
        elif self.session_running and self.session_paused:
            elapsed = int(self.elapsed_before_pause)
        else:
            elapsed = 0

        mins = elapsed // 60
        secs = elapsed % 60
        self.timer_label.configure(text=f"Session Time: {mins:02}:{secs:02}")

        self.app.after(200, self.update_timer)

    def update_punch_label(self):
        if self.session_running and not self.session_paused:
            self.punch_label.configure(text=self.latest_punch)
        elif not self.session_running:
            self.punch_label.configure(text="WAITING...")

        self.app.after(50, self.update_punch_label)

    def update_history(self):
        if self.recent_punches:
            text = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(self.recent_punches))
        else:
            text = "No punches yet"

        self.history_label.configure(text=text)
        self.app.after(100, self.update_history)

    def update_sound(self):
        if self.pending_ko_sound:
            self.pending_ko_sound = False
            self.play_ko_sound()

        self.app.after(50, self.update_sound)

    def run(self):
        self.app.mainloop()


if __name__ == "__main__":
    ui = PunchDetectorUI()
    ui.run()