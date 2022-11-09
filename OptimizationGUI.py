from cgitb import text
from inspect import getcoroutinelocals
from platform import python_revision
import numpy as np
from OptimizationMethods import *
from tkinter import *
import random

def getLocation(w, h, x, y, m):
    x_new = x/m*(w/2) + w/2
    y_new = h/2 - y/m*(h/2)
    return x_new, y_new

def getCoordinates(w, h, x, y, m):
    c_x = (2*x*m)/w-m
    c_y = m-(2*m*y)/h
    return c_x, c_y

wi = 1200; he = 800
max = 20
st = np.array([max/2,max/2])
optimize = False

def drawStartpoint(canvas):
    x, y = getLocation(wi*2/3, he, st[0],st[1], max)
    return canvas.create_oval(x-2,y-2,x+2,y+2, tags="start")

def createGrid(canvas, width, height, max):
    canvas.create_line(10,height/2,width-10,height/2, arrow=LAST)
    canvas.create_line(width/2,10,width/2,height-10, arrow=FIRST)
    for i in range(max-1):
        x_n, y_n = getLocation(width, height, 0, i, max)
        canvas.create_line(x_n-2,y_n,x_n+2,y_n)
        canvas.create_line(x_n-2,height/2+y_n,x_n+2,height/2+y_n)
        canvas.create_line(y_n,x_n-2,y_n,x_n+2)
        canvas.create_line(width/2+y_n,x_n-2,width/2+y_n,x_n+2)

def drawIteration(canvas,start,end, width, height, max, color):
    x_s, y_s = getLocation(width, height, start[0],start[1],max)
    x_e, y_e = getLocation(width, height, end[0],end[1],max)
    canvas.create_line(x_s,y_s,x_e,y_e,arrow=LAST,fill=color,tags="its")

def setStartposition(event):
    global st
    global stpo
    x, y = event.x,event.y
    if x > wi * 2/3:
        return
    x_grid, y_grid = getCoordinates(wi*2/3, he, x, y, max)
    st = np.array([x_grid,y_grid])
    window.delete("start")
    stpo = drawStartpoint(window)

def startOptimization(event):
    x_d = x_der.get()
    y_d = y_der.get()
    der = lambda z: np.array([(lambda x: eval(x_d))(z[0]),(lambda y: eval(y_d))(z[1])])
    sder = lambda x: der(x) + np.random.normal(loc=np.array([0,0]),scale=2)
    window.delete("its")
    global optimize, gd_it, sm_it, rp_it, ad_it, st
    optimize = True
    if gd_op.get():
        gd_it = gradient_descend(sder, 1, st, 2)
    if sm_op.get():
        sm_it = sgd_momentum(sder, 1, st, 2)
    if rp_op.get():
        rp_it = rmsprop(sder, 1, st, 2)
    if ad_op.get():
        ad_it = adam(sder, 0.5, st, 2)


master = Tk(className=" Optimization Visualization")
master.configure(bg="black")
window  = Canvas(master, width=wi, height=he)
window.bind('<Button-1>',setStartposition)
window.pack()

createGrid(window,wi*2/3,he,max)

gd_it = []; sm_it = []; rp_it = []; ad_it = []
gd_op = BooleanVar(); sm_op = BooleanVar(); rp_op = BooleanVar(); ad_op = BooleanVar()
gd_op.set(False); sm_op.set(False); rp_op.set(False); ad_op.set(False)
gr = lambda x: 2*x

opbutton = Button(window,text="Optimize",width = 10)
opbutton.place(x=wi*5/6-25,y=he/2)
opbutton.bind('<Button-1>',startOptimization)
gdcb = Checkbutton(window, text="Stochastic Gradient Descent", variable = gd_op, fg="light blue")
gdcb.place(x=wi*5/6-25,y=40)
agcb = Checkbutton(window, text="SGD with Momentum", variable = sm_op, fg="red")
agcb.place(x=wi*5/6-25,y=80)
rpcb = Checkbutton(window, text="RMSProp", variable = rp_op, fg="light green")
rpcb.place(x=wi*5/6-25,y=120)
adcb = Checkbutton(window, text="Adam", variable = ad_op,fg="orange")
adcb.place(x=wi*5/6-25,y=160)
x_der = Entry(window)
x_der.place(x=wi*5/6-25, y=he/2+40)
y_der = Entry(window)
y_der.place(x=wi*5/6-25, y=he/2+80)
sc = Scale(window, orient=HORIZONTAL,from_=1,to_=10)
sc.place(x=wi*5/6-25, y=he/2+200)
stpo = drawStartpoint(window)

def mloop():
    if optimize:
        if gd_op.get() and len(gd_it) > 1:
            drawIteration(window,gd_it[0],gd_it[1],wi*2/3,he,max,'light blue')
            gd_it.pop(0)
        if sm_op.get() and len(sm_it) > 1:
            drawIteration(window,sm_it[0],sm_it[1],wi*2/3,he,max,'red')
            sm_it.pop(0)
        if rp_op.get() and len(rp_it) > 1:
            drawIteration(window,rp_it[0],rp_it[1],wi*2/3,he,max,'green')
            rp_it.pop(0)
        if ad_op.get() and len(ad_it) > 1:
            drawIteration(window,ad_it[0],ad_it[1],wi*2/3,he,max,'orange')
            ad_it.pop(0)
    master.after(int(1000/sc.get()),mloop)

master.after(int(1000/sc.get()), mloop)
master.mainloop()