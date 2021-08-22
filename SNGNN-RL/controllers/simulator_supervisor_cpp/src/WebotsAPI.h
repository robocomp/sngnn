#pragma once

#include <webots/Supervisor.hpp>
#include <WorldEntity.h>
#include <Wall.h>
#include <Goal.h>
#include <Obstacle.h>
#include <Human.h>
#include <Relation.h>

using namespace webots;




class Wall;
class RobotNode;

class WebotsAPI : public Supervisor
{
public:
    WebotsAPI();
    void close();

    Wall *create_wall(const std::string &name, const double &x1, const double &z1, const double &x2, const double &z2);

    Node *create_floor(const std::string &name, double x, double z, double width, double length);

    Goal *create_goal(const double &x, const double &z);

    Obstacle *create_object(const std::string &name, const double &x, const double &z, const double &angle, const std::string &otype);

    RobotNode *create_robot(const double &x, const double &z, const double &angle);

    Human *create_humans(const std::string &name, const double &x, const double &z, const double &angle, const std::string &command);

    Relation *create_relations(const std::string &name, const std::string &relation_type, WorldEntity *node1, WorldEntity *node2);


    void remove_objects();



// protected:
    Node *root;
    Field *rootChildren;
};


    // def create_relation(self, name, relation_type, node1, node2):
    //     return Relation(name, relation_type, node1, node2, self)

    // def create_human(self, name, x, y, orient, command=None):
    //     base_str = "".join(open("../../protos/pedestrian_base.wbo", "r").readlines())
    //     string = base_str.replace("SEEDNAME", name)
    //     string = string.replace("ORIENT", str(orient))
    //     string = string.replace("SEEDX", str(x))
    //     string = string.replace("SEEDZ", str(y))
    //     this->rootchildren->importMFNodeFromString(-1, string)
    //     human_handle = this->getFromDef(name)
    //     human = Human(human_handle)
    //     human.move(command)
    //     return human




// class Relation(Entity):
//     def __init__(self, name, relation_type, node1, node2, supervisor):
//         super(Relation, self).__init__()
//         this->relation_type = relation_type
//         this->node1, this->node2 = node1, node2
//         this->p1 = [*this->node1.get_position()]
//         this->p2 = [*this->node2.get_position()]

//         x, z = 0.5 * (this->p1[0] + this->p2[0]), 0.5 * (this->p1[1] + this->p2[1])
//         this->length = np.linalg.norm(np.array(this->p2) - np.array(this->p1))
//         cx = (this->p2[0] - this->p1[0]) / this->length
//         cz = (this->p2[1] - this->p1[1]) / this->length
//         if cz != 0:
//             za = -cx / cz
//             xa = 1
//         else:
//             za = 1
//             xa = 0
//         this->name = name
//         this->supervisor = supervisor
//         this->rootChildren = wapi->getRoot()->getField("children")

//         base_str = "".join(open("../../protos/relation_base.wbo", "r").readlines())
//         sx = this->length
//         string = base_str.replace("SEEDNAME", this->name)
//         string = string.replace("SEEDX", str(sx))

//         this->rootchildren->importMFNodeFromString(-1, string)
//         relation = this->wapi->getFromDef(this->name)
//         y = relation->getField("translation")->getSFVec3f()[1]
//         relation->getField("translation")->setSFVec3f([x, y, z])
//         relation->getField("rotation")->setSFRotation([xa, 0, za, pi / 2])

//         this->handle = relation

//     def get_handle(self):
//         return this->handle._handle

//     def remove(self):
//         this->handle.remove()

//     def move(self):
//         this->p1 = [*this->node1.get_position()]
//         this->p2 = [*this->node2.get_position()]

//         x, z = 0.5 * (this->p1[0] + this->p2[0]), 0.5 * (this->p1[1] + this->p2[1])
//         this->length = np.linalg.norm(np.array(this->p2) - np.array(this->p1))
//         cx = (this->p2[0] - this->p1[0]) / this->length
//         cz = (this->p2[1] - this->p1[1]) / this->length
//         if cz != 0:
//             za = -cx / cz
//             xa = 1
//         else:
//             za = 1
//             xa = 0
//         y = this->handle->getField("translation")->getSFVec3f()[1]
//         this->handle->getField("translation")->setSFVec3f([x, y, z])
//         this->handle->getField("rotation")->setSFRotation([xa, 0, za, pi / 2])
//         this->handle->getField("children").getMFNode(0)->getField(
//             "geometry"
//         )->getSFNode()->getField("height")->setSFFloat(this->length)


// class Human(Entity):
//     def __init__(self, handle):
//         super(Human, self).__init__()
//         this->handle = handle

//     def set_id(self, id):
//         this->id = id

//     def get_id(self):
//         return this->id

//     def set_position(self, position):
//         x, z = position
//         y = this->handle->getField("translation")->getSFVec3f()[1]
//         this->handle->getField("translation")->setSFVec3f([x, y, z])

//     def set_orientation(self, angle):
//         this->handle->getField("rotation")->setSFRotation([0, 1, 0, angle])

//     def move(self, command):
//         if command is None:
//             pass
//         else:
//             controller_arguments = this->handle->getField("controllerArgs")
//             trajectory, speed = command
//             generator = [(str(p[0]) + " " + str(p[1])) for p in trajectory]
//             trajectory_str = "--trajectory="
//             for i in generator:
//                 trajectory_str += i + ","
//             trajectory_str = trajectory_str[:-1]
//             controller_arguments.insertMFString(0, trajectory_str)
//             controller_arguments.insertMFString(1, "--speed=" + str(speed))

//     def stop(self):
//         controller_arguments = this->handle->getField("controllerArgs")
//         controller_arguments.removeMF(0)
//         controller_arguments.removeMF(1)

//     def get_handle(self):
//         return this->handle

//     def get_velocity(self):
//         return this->handle.getVelocity()

//     def remove(self):
//         this->handle.remove()

