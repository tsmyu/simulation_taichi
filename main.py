import taichi as ti

from simulator import Simulator

def main():
    # ti.init(arch=ti.cpu)
    # ti.init(arch=ti.gpu, device_memory_GB=2.0)
    arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
    ti.init(arch=arch)

    resolution = (400, 200)
    window = ti.ui.Window("Fluid Simulation", resolution, vsync=False)
    canvas = window.get_canvas()

    dt = 0.01
    density = 1.166 #air at 20 
    velocity = 344 #air at about 20 
    sim = Simulator.create(resolution[1], dt, density, velocity)
    # video_manager = ti.tools.VideoManager(output_dir=str(img_path), framerate=30, automatic_build=False)
    # count = 0
    paused = False
    while window.running:
        if not paused:
            sim.step()

        img = sim.get_norm_field()

        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.ui.ESCAPE:
                break
            elif e.key == "p":
                paused = not paused

        canvas.set_image(img)
        window.show()

        # if count % 20 == 0:
        #     video_manager.write_frame(img)
        # count += 1

    # video_manager.make_video(mp4=True)


if __name__ == "__main__":
    main()