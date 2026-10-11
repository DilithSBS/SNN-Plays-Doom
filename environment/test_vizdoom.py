from vizdoom import DoomGame

game = DoomGame()

game.load_config("configs/basic.cfg")
game.init()

print("ViZDoom started!")

game.new_episode()

for step in range(10):
    state = game.get_state()

    if state is not None:
        print(
            f"Step {step}: "
            f"screen shape = {state.screen_buffer.shape}"
        )

    reward = game.make_action([1, 0, 0])
    print("Reward:", reward)

game.close()

print("ViZDoom closed!")