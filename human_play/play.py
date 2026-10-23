#!/usr/bin/env python3
"""
Interactive Human Play for ViZDoom
==================================
Allows human players to see and play Doom in a real-time graphical window
using intuitive keyboard (WASD / Arrows) and mouse controls.

Usage:
    python human_play/play.py --scenario basic
    python human_play/play.py --scenario my_way_home
    python human_play/play.py --scenario defend_the_center
    python human_play/play.py --res 800x600 --episodes 5
    python human_play/play.py --record gameplay.mp4
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import vizdoom as vzd


RESOLUTION_MAP = {
    "320x240": vzd.ScreenResolution.RES_320X240,
    "640x480": vzd.ScreenResolution.RES_640X480,
    "800x600": vzd.ScreenResolution.RES_800X600,
    "1024x768": vzd.ScreenResolution.RES_1024X768,
    "1280x720": vzd.ScreenResolution.RES_1280X720,
    "1920x1080": vzd.ScreenResolution.RES_1920X1080,
}

PRESET_SCENARIOS = {
    "basic": {
        "config": PROJECT_ROOT / "human_play" / "configs" / "human_basic.cfg",
        "description": "Combat Arena: Strafe and shoot the Cacodemon monster",
        "space_action": "+attack",
    },
    "my_way_home": {
        "config": PROJECT_ROOT / "human_play" / "configs" / "human_my_way_home.cfg",
        "description": "3D Labyrinth: Navigate multi-room corridors to find the green armor",
        "space_action": "+use",
    },
    "defend_the_center": {
        "config": PROJECT_ROOT / "human_play" / "configs" / "human_defend_the_center.cfg",
        "description": "Wave Defense: Stand in center, pivot 360°, eliminate monster waves",
        "space_action": "+attack",
    },
}


def print_banner(scenario_name: str, desc: str, res_str: str, sound_enabled: bool, episodes: int):
    sound_status = "ENABLED [Audio ON]" if sound_enabled else "DISABLED [Run with --sound to enable]"
    ep_str = f"{episodes} Rounds" if episodes > 0 else "Endless (Press Ctrl+C to quit)"
    
    print("=" * 78)
    print("              🎮  DOOM INTERACTIVE HUMAN PLAY (KEYBOARD & MOUSE)  🎮")
    print("=" * 78)
    print(f"  Scenario     : {scenario_name.upper()} ({desc})")
    print(f"  Resolution   : {res_str} Full Color (RGB24)")
    print(f"  Audio        : {sound_status}")
    print(f"  Game Rounds  : {ep_str}")
    print("-" * 78)
    print("  ⌨️  KEYBOARD CONTROLS:")
    print("      Movement  :  W / S / Up / Down      --> Move Forward / Backward")
    print("      Strafing  :  A / D                  --> Strafe Left / Right")
    print("      Turning   :  Mouse / Left / Right   --> Look / Aim Left / Right")
    print("      Fire Gun  :  Left Mouse Click / Ctrl / Space")
    print("      Open/Use  :  E / Enter (Open doors, flip switches)")
    print("      Sprint    :  Left Shift (Hold to run)")
    print("      Weapons   :  1 - 7 (Select weapon)")
    print("      Pause     :  ESC or Pause key")
    print("      Quit Game :  Close the window, or press Ctrl+C in terminal")
    print("=" * 78)
    print()


def configure_keybindings(game: vzd.DoomGame, space_action: str = "+attack"):
    """
    Sends console bind commands directly to the underlying ZDoom engine
    to ensure full WASD, Arrow keys, Mouse, and Action key support.
    """
    commands = [
        # WASD Movement
        "bind w +forward",
        "bind s +back",
        "bind a +moveleft",
        "bind d +moveright",
        # Arrow Keys
        "bind uparrow +forward",
        "bind downarrow +back",
        "bind leftarrow +left",
        "bind rightarrow +right",
        # Combat & Actions
        "bind mouse1 +attack",
        "bind ctrl +attack",
        f"bind space {space_action}",
        "bind e +use",
        "bind enter +use",
        "bind shift +speed",
        "bind alt +strafe",
        # Weapon Switching
        "bind 1 'slot 1'",
        "bind 2 'slot 2'",
        "bind 3 'slot 3'",
        "bind 4 'slot 4'",
        "bind 5 'slot 5'",
        "bind 6 'slot 6'",
        "bind 7 'slot 7'",
    ]
    for cmd in commands:
        try:
            game.send_game_command(cmd)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Play ViZDoom interactively with keyboard & mouse in real time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="basic",
        help="Scenario to play: 'basic', 'my_way_home', 'defend_the_center', or custom .cfg path",
    )
    parser.add_argument(
        "--res",
        type=str,
        default="800x600",
        choices=list(RESOLUTION_MAP.keys()),
        help="Window screen resolution",
    )
    parser.add_argument(
        "--sound",
        action="store_true",
        default=False,
        help="Enable game sound/music (Note: in WSL2, ensure PulseAudio/WSLg audio is active)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to play (0 for infinite loop until quit)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=35,
        help="Target framerate (35 is authentic native Doom ticrate)",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Optional path to output MP4 video recording of your gameplay session",
    )

    args = parser.parse_args()

    # Determine config file path
    scenario_key = args.scenario.lower().strip()
    if scenario_key in PRESET_SCENARIOS:
        preset = PRESET_SCENARIOS[scenario_key]
        cfg_path = preset["config"]
        desc = preset["description"]
        space_act = preset["space_action"]
    else:
        # Check if it's a file path or a scenario in configs/ or vizdoom package
        custom_candidate = Path(args.scenario)
        if custom_candidate.exists():
            cfg_path = custom_candidate
            desc = "Custom User Scenario"
            space_act = "+attack"
        else:
            configs_dir_candidate = PROJECT_ROOT / "configs" / f"{args.scenario}.cfg"
            if configs_dir_candidate.exists():
                cfg_path = configs_dir_candidate
                desc = f"Project scenario: {args.scenario}"
                space_act = "+attack"
            else:
                print(f"[Error] Scenario configuration '{args.scenario}' not found!")
                print(f"Available presets: {list(PRESET_SCENARIOS.keys())}")
                sys.exit(1)

    print_banner(scenario_key, desc, args.res, args.sound, args.episodes)

    # Initialize DoomGame instance
    game = vzd.DoomGame()
    game.load_config(str(cfg_path))

    # Apply graphical and window overrides
    game.set_screen_resolution(RESOLUTION_MAP[args.res])
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(True)
    game.set_render_hud(True)
    game.set_render_crosshair(True)
    game.set_render_weapon(True)
    game.set_render_decals(True)
    game.set_render_particles(True)
    game.set_sound_enabled(args.sound)

    # Set interactive spectator mode
    game.set_mode(vzd.Mode.ASYNC_SPECTATOR)

    # Enable full button set for the human player
    all_buttons = [
        vzd.Button.MOVE_LEFT,
        vzd.Button.MOVE_RIGHT,
        vzd.Button.ATTACK,
        vzd.Button.MOVE_FORWARD,
        vzd.Button.MOVE_BACKWARD,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
        vzd.Button.USE,
        vzd.Button.SPEED,
        vzd.Button.STRAFE,
    ]
    game.set_available_buttons(all_buttons)

    try:
        game.init()
    except Exception as e:
        print(f"[Error] Failed to initialize ViZDoom game window: {e}")
        print("Tip for WSL2 / Linux: Make sure DISPLAY or WSLg is running.")
        sys.exit(1)

    # Inject WASD and custom keyboard bindings
    configure_keybindings(game, space_action=space_act)

    # Optional video recorder setup
    video_writer = None
    if args.record:
        try:
            import cv2
            out_path = Path(args.record).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            w, h = (int(args.res.split("x")[0]), int(args.res.split("x")[1]))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))
            print(f"🎥 Recording gameplay session to: {out_path}")
        except Exception as err:
            print(f"[Warning] Failed to initialize video recorder: {err}")
            video_writer = None

    ep_count = 0
    max_episodes = args.episodes if args.episodes > 0 else float("inf")
    session_scores = []
    session_kills = []

    try:
        while ep_count < max_episodes:
            ep_count += 1
            print(f"\n>>> Starting Round #{ep_count} ... (Get Ready!)")
            game.new_episode()
            
            # Re-verify keybindings at start of episode
            configure_keybindings(game, space_action=space_act)

            start_time = time.time()
            ticks = 0

            while not game.is_episode_finished():
                game.advance_action()
                ticks += 1

                # Capture frame for recording if requested
                if video_writer is not None:
                    state = game.get_state()
                    if state is not None and state.screen_buffer is not None:
                        # Convert RGB to BGR for OpenCV
                        bgr_frame = state.screen_buffer[:, :, ::-1]
                        video_writer.write(bgr_frame)

            # Round finished
            duration = time.time() - start_time
            total_reward = game.get_total_reward()
            session_scores.append(total_reward)

            # Extract final game variables
            last_state = game.get_state()
            health = int(last_state.game_variables[vzd.GameVariable.HEALTH]) if last_state and vzd.GameVariable.HEALTH in game.get_available_game_variables() else "N/A"
            kills = int(last_state.game_variables[vzd.GameVariable.KILLCOUNT]) if last_state and vzd.GameVariable.KILLCOUNT in game.get_available_game_variables() else 0
            session_kills.append(kills)

            # Determine round status
            if scenario_key == "basic":
                result_str = "🏆 MONSTER ELIMINATED! EXCELLENT SHOT!" if total_reward > 0 else "💀 TIME UP / MISSED"
            elif scenario_key == "my_way_home":
                result_str = "🏆 GOAL REACHED! ARMOR COLLECTED!" if total_reward > 50 else "🧭 LABYRINTH TIME UP"
            else:
                result_str = "ROUND COMPLETE"

            print("-" * 65)
            print(f"🏁  Round #{ep_count} Result: {result_str}")
            print(f"    • Total Score / Reward : {total_reward:+.1f}")
            print(f"    • Survival Time        : {duration:.1f}s ({ticks} game ticks)")
            if health != "N/A":
                print(f"    • Final Health         : {health}%")
            if kills > 0:
                print(f"    • Monsters Killed      : {kills}")
            print("-" * 65)

            if ep_count < max_episodes:
                print("Starting next round in 2 seconds... (or press Ctrl+C in terminal to exit)")
                time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n\n[Interrupted] Gameplay paused by user.")

    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"🎥 Video saved to: {args.record}")
        
        try:
            game.close()
        except Exception:
            pass

        print("\n" + "=" * 65)
        print("                  SESSION SUMMARY")
        print("=" * 65)
        print(f"  Rounds Played : {len(session_scores)}")
        if session_scores:
            print(f"  Best Score    : {max(session_scores):+.1f}")
            print(f"  Average Score : {sum(session_scores)/len(session_scores):+.1f}")
            if any(k > 0 for k in session_kills):
                print(f"  Total Kills   : {sum(session_kills)}")
        print("  Thanks for playing SNN-Plays-Doom!")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
