from sndgAPI import *

sn = SNScenario()
sn.add_room({'x0':100,'y0':100,'x1':-100,'y1':100,'x2':-100,'y2':-100,'x3':100,'y3':-100})
sn.add_human(Human(1,0,0,10))
sn.add_object(Object(2,0,10,10))
sn.add_interaction([1,2])
sngnn = SNGNN()
print(sngnn.predict(sn))

