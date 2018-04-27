def cross_block(xl, x0, units, name):
    w = sym.Variable(name+"_w")
    b = sym.Variable(name+"_b")
    fc = sym.dense(data=xl, weight=w, user_bais=False,
                   units=units, name=name+'_fc1')
    fc = sym.dense(data=fc, weight=x0, bias=b, units=units, name=name+'_fc2')
    eadd = sym.elemwise_add(lhs=fc, rhs=xl, name=name+'_eadd1')
    return fc
