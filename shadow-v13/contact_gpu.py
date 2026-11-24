import warp as wp
@wp.kernel(enable_backward=False)
def reduce_contacts(nacon:wp.array(dtype=wp.int32),geom:wp.array(dtype=wp.vec2i),
                    world:wp.array(dtype=wp.int32),f:wp.array(dtype=wp.spatial_vector),
                    mapping:wp.array(dtype=wp.int32),ball:int,floor:int,
                    per_digit:wp.array2d(dtype=wp.float32),total:wp.array(dtype=wp.float32)):
    i=wp.tid()
    if i>=nacon[0]:return
    g=geom[i];other=int(-1)
    if g[0]==ball:other=g[1]
    elif g[1]==ball:other=g[0]
    if other<0 or other==floor:return
    w=world[i];value=wp.max(0.0,f[i][0])
    wp.atomic_add(total,w,value)
    digit=mapping[other]
    if digit>=0:wp.atomic_add(per_digit,w,digit,value)


@wp.kernel(enable_backward=False)
def penetration(nacon:wp.array(dtype=wp.int32),geom:wp.array(dtype=wp.vec2i),
                world:wp.array(dtype=wp.int32),dist:wp.array(dtype=wp.float32),
                ball:int,floor:int,self_depth:wp.array(dtype=wp.float32),object_depth:wp.array(dtype=wp.float32)):
    i=wp.tid()
    if i>=nacon[0]:return
    g=geom[i]
    if g[0]==floor or g[1]==floor:return
    dep=wp.max(0.0,-dist[i])
    if g[0]==ball or g[1]==ball:
        wp.atomic_max(object_depth,world[i],dep)
    else:
        wp.atomic_max(self_depth,world[i],dep)

@wp.kernel(enable_backward=False)
def reduce_opposition(nacon:wp.array(dtype=wp.int32),geom:wp.array(dtype=wp.vec2i),
    world:wp.array(dtype=wp.int32),frame:wp.array(dtype=wp.mat33),
    force:wp.array(dtype=wp.spatial_vector),xmat:wp.array2d(dtype=wp.mat33),
    mapping:wp.array(dtype=wp.int32),ball:int,oid:int,
    normals:wp.array3d(dtype=wp.float32),loads:wp.array2d(dtype=wp.float32)):
    i=wp.tid()
    if i>=nacon[0]:return
    g=geom[i];other=int(-1);sign=float(1.0)
    if g[0]==ball:other=g[1]
    elif g[1]==ball:
        other=g[0];sign=-1.0
    if other<0:return
    digit=mapping[other]
    if digit<0:return
    f=wp.max(0.0,force[i][0])
    if f<0.005:return
    w=world[i];rot=xmat[w,oid]
    axis=wp.vec3(rot[0,2],rot[1,2],rot[2,2])
    fr=frame[i];n=sign*wp.vec3(fr[0,0],fr[0,1],fr[0,2])
    radial=n-wp.dot(n,axis)*axis
    group=int(0)
    if digit==4:group=1
    wp.atomic_add(loads,w,group,f*wp.length(radial))
    for j in range(3):wp.atomic_add(normals,w,group,j,f*radial[j])

