import taichi as ti
import taichi_glsl as ts
import time
ti.init(arch=ti.gpu)

asp = 16/9
h = 600
w = int(asp*h)
res = w,h
resf = ts.vec2(float(w), float(h))

pixels = ti.Vector.field(3,dtype=ti.f32,shape=res)

@ti.func
def rot(a):
    c = ti.cos(a)
    s = ti.sin(a)
    return ts.mat([c,-s],[s,c])

@ti.kernel
def render(t:ti.f32):
    # col0 = ts.vec3(255.,131.,137.)/255.0

    for fragCoord in ti.grouped(pixels):  # Стандартизуем понятие координат
        uv =(fragCoord - 0.5*resf) / resf[1] # координаты внутри векторного поля
        m = rot(0.1 * t)
        uv = m @ uv
        uv *= 10.0
        fuv = ts.fract(uv) - 0.5

        grid = ts.smoothstep(ti.abs(fuv).max(), 0.4, 0.5)

        col = ts.vec3(1., 2., 0.)
        col = ts.mix(
            col,
            ts.vec3(1.0),
            grid
        )

        col += ts.vec3(1.,2.,0.) * grid
        # col.gb = uv + 0.5 # ы хотим взять из вектора две координаты и в определенном порядке. Прикол шейдера

        pixels[fragCoord] = ts.clamp(col,0.,1.) # clamp(col**(1/2.2), 0., 1. )



#
# @ti.kernel
# gef render(t:ti.f32, frame:ti.int):
#
#



if __name__ == '__main__':

    gui = ti.GUI("Taichi basic shader", res=res, fast_gui = True)
    start = time.time()

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break
        delta_time = time.time() - start

        render(delta_time)
        gui.set_image(pixels)
        gui.show()
    gui.close()