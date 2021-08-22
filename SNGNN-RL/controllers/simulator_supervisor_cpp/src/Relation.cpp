// class Relation(Entity):
//     def __init__(self, name, relation_type, node1, node2, supervisor):
//         super(Relation, self).__init__()
//         self.relation_type = relation_type
//         self.node1, self.node2 = node1, node2
//         self.p1 = [*self.node1.get_position()]
//         self.p2 = [*self.node2.get_position()]

//         x, z = 0.5 * (self.p1[0] + self.p2[0]), 0.5 * (self.p1[1] + self.p2[1])
//         self.length = np.linalg.norm(np.array(self.p2) - np.array(self.p1))
//         cx = (self.p2[0] - self.p1[0]) / self.length
//         cz = (self.p2[1] - self.p1[1]) / self.length
//         if cz != 0:
//             za = -cx / cz
//             xa = 1
//         else:
//             za = 1
//             xa = 0
//         self.name = name
//         self.supervisor = supervisor
//         self.rootChildren = supervisor.getRoot().getField("children")

//         base_str = "".join(open("../../protos/relation_base.wbo", "r").readlines())
//         sx = self.length
//         string = base_str.replace("SEEDNAME", self.name)
//         string = string.replace("SEEDX", str(sx))

//         self.rootChildren.importMFNodeFromString(-3, string)
//         relation = self.supervisor.getFromDef(self.name)
//         y = relation.getField("translation").getSFVec3f()[1]
//         relation.getField("translation").setSFVec3f([x, y, z])
//         relation.getField("rotation").setSFRotation([xa, 0, za, pi / 2])

//         self.handle = relation

//     def get_handle(self):
//         return self.handle._handle

//     def remove(self):
//         self.handle.remove()

//     def move(self):
//         self.p1 = [*self.node1.get_position()]
//         self.p2 = [*self.node2.get_position()]

//         x, z = 0.5 * (self.p1[0] + self.p2[0]), 0.5 * (self.p1[1] + self.p2[1])
//         self.length = np.linalg.norm(np.array(self.p2) - np.array(self.p1))
//         cx = (self.p2[0] - self.p1[0]) / self.length
//         cz = (self.p2[1] - self.p1[1]) / self.length
//         if cz != 0:
//             za = -cx / cz
//             xa = 1
//         else:
//             za = 1
//             xa = 0
//         y = self.handle.getField("translation").getSFVec3f()[1]
//         self.handle.getField("translation").setSFVec3f([x, y, z])
//         self.handle.getField("rotation").setSFRotation([xa, 0, za, pi / 2])
//         self.handle.getField("children").getMFNode(0).getField(
//             "geometry"
//         ).getSFNode().getField("height").setSFFloat(self.length)




#include <fstream>
#include <iostream>
#include <boost/algorithm/string.hpp>

#include <WebotsAPI.h>
#include <Relation.h>
#include <math.h>
#include <cmath>

Relation::Relation(const std::string &name, const std::string &relation_type, WorldEntity *node1, WorldEntity *node2, WebotsAPI *wapi): WorldEntity(wapi)
{
    this->dynamic = false;

    this->x = (node1->x + node2->x) / 2.0;
    this->z = (node1->z + node2->z) / 2.0;

    float length = sqrt(pow(node2->x - node1->x,2) + pow(node2->z - node1->z,2));
    float cx = (node2->x - node1->x) / length;
    float cz = (node2->z - node1->z) / length;

    float za, xa;
    if (cz != 0)
    {
        xa = 1.0;
        za = -(cx/cz);
    }
    else
    {
        za = 1.0;
        xa = 0.0;
    }

    std::string base_str;
    std::vector<double> rotV = {xa, 0, za, M_PI / 2};

    base_str = "../../protos/relation_base.wbo";


    std::ifstream ifs(base_str);
    std::string content( (std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()) );
    boost::replace_all(content, "SEEDNAME", name);
    boost::replace_all(content, "SEEDX", std::to_string(length));

    wapi->rootChildren->importMFNodeFromString(-1, content);
    this->handle = this->wapi->getFromDef(name);
    double y = this->handle->getField("translation")->getSFVec3f()[1];
    const double posV[3] = {this->x, y, this->z};
    this->handle->getField("translation")->setSFVec3f(posV);

    this->handle->getField("rotation")->setSFRotation(&rotV[0]);

}