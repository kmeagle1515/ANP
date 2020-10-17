import math

# Input points from approx poly
pts = [(10,20),(120,22),(300,18),(310,100),(150,100),(8,95)]

print("input points" , pts)

def distance(pt1,pt2):
    return math.sqrt( (pt2[0]-pt1[0])*(pt2[0]-pt1[0]) + (pt2[1]-pt1[1])*(pt2[1]-pt1[1]) )


# width and height of cropped image
w = 300
h = 100

#starting edge points as centre
top_left_pt = (w/2,h/2)
top_right_pt = (w/2,h/2)
bottom_right_pt = (w/2,h/2)
botton_left_pt = (w/2,h/2)

#assign highest dimension width or height ( Mostly width)
d_tl = w
d_tr = w
d_bl = w
d_br = w

for pt in pts:
    temp_d_tl = distance(pt,(0,0))
    temp_d_tr = distance(pt,(w,0))
    temp_d_bl = distance(pt,(0,h))
    temp_d_br = distance(pt,(w,h))
    
    # compare the points against edge points to get edge points
    if( temp_d_tl < d_tl ):
        d_tl = temp_d_tl
        top_left_pt = pt

    if( temp_d_tr < d_tr ):
        d_tr = temp_d_tr
        top_right_pt = pt
    
    if( temp_d_bl < d_bl ):
        d_bl = temp_d_bl
        botton_left_pt = pt
    
    if( temp_d_br < d_br ):
        d_br = temp_d_br
        bottom_right_pt = pt

Edge_points = [top_left_pt,top_right_pt,bottom_right_pt,botton_left_pt]

print("Edge Points",Edge_points)