# 🎮 Interactive Human Play: Play ViZDoom with Keyboard & Mouse

This directory provides an interactive, real-time environment allowing you to **see, play, and test Doom directly using your keyboard and mouse**.

---

## ⚡ Quickstart

From the repository root (`SNN-Plays-Doom/`):

### 1. Launch Combat Arena (`basic.cfg`)
```bash
# Using the launcher script:
./human_play/play.sh

# Or directly using Python:
python human_play/play.py
```
*Objective*: Strafe left/right, aim, and shoot down the flying Cacodemon!

---

### 2. Launch 3D Labyrinth Traversal (`my_way_home.cfg`)
```bash
./human_play/play.sh --scenario my_way_home
```
*Objective*: Navigate through multi-colored corridors and rooms to locate the green armor vest.

---

### 3. Launch 360° Monster Wave Defense (`defend_the_center.cfg`)
```bash
./human_play/play.sh --scenario defend_the_center
```
*Objective*: Stand in the center of the circular arena, pivot 360°, and eliminate incoming waves of monsters!

---

## ⌨️ Controls Reference

| Action | Primary Key | Secondary / Alternative |
|---|---|---|
| **Move Forward** | `W` | `Up Arrow` |
| **Move Backward** | `S` | `Down Arrow` |
| **Strafe Left** | `A` | `,` (Comma) |
| **Strafe Right** | `D` | `.` (Period) |
| **Turn / Aim Left** | `Mouse Left` | `Left Arrow` |
| **Turn / Aim Right** | `Mouse Right` | `Right Arrow` |
| **Fire Weapon** | `Left Mouse Click` | `Left Ctrl` or `Space` |
| **Open Doors / Use Switch** | `E` | `Enter` or `Space` (in navigation) |
| **Sprint / Run** | `Left Shift` | `Caps Lock` (toggle) |
| **Switch Weapons** | `1` – `7` | Mouse Wheel Up / Down |
| **Pause Game** | `ESC` | `Pause` key |
| **Quit Game** | Close Window | `Ctrl + C` in Terminal |

---

## ⚙️ Options & Arguments

You can customize your experience using flags:

| Flag | Default | Description | Example |
|---|---|---|---|
| `--scenario` | `basic` | Scenario to launch (`basic`, `my_way_home`, `defend_the_center`, or custom `.cfg` path) | `--scenario my_way_home` |
| `--res` | `800x600` | Window resolution: `640x480`, `800x600`, `1024x768`, `1280x720`, `1920x1080` | `--res 1024x768` |
| `--sound` | `False` | Enable in-game sound effects & music | `--sound` |
| `--episodes` | `10` | Number of rounds to play (`0` for endless) | `--episodes 0` |
| `--fps` | `35` | Framerate limit (35 is authentic Doom engine ticrate) | `--fps 35` |
| `--record` | `None` | Save your gameplay session directly to an MP4 video | `--record gameplay.mp4` |

### Examples:
```bash
# Play endless rounds in high-res 1024x768:
./human_play/play.sh --scenario basic --res 1024x768 --episodes 0

# Record your navigation run to an MP4 video:
./human_play/play.sh --scenario my_way_home --record my_run.mp4
```

---

## 🐧 Notes for WSL2 & Linux Users

- **WSLg (Windows 11 / Windows 10)**: ViZDoom windows render natively via hardware acceleration. If the window does not appear, ensure `DISPLAY=:0` is exported in your terminal.
- **Headless SSH**: If playing over remote SSH, enable X11 forwarding (`ssh -X` or `ssh -Y`) or use Moonlight / VNC / WSLg desktop.
