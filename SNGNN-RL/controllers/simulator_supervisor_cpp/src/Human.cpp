// class Human(Entity):
//     def __init__(self, handle):
//         super(Human, self).__init__()
//         self.handle = handle

//     def set_id(self, id):
//         self.id = id

//     def get_id(self):
//         return self.id

//     def set_position(self, position):
//         x, z = position
//         y = self.handle.getField("translation").getSFVec3f()[1]
//         self.handle.getField("translation").setSFVec3f([x, y, z])

//     def set_orientation(self, angle):
//         self.handle.getField("rotation").setSFRotation([0, 1, 0, angle])

//     def move(self, command):
//         if command is None:
//             pass
//         else:
//             controller_arguments = self.handle.getField("controllerArgs")
//             trajectory, speed = command
//             generator = [(str(p[0]) + " " + str(p[1])) for p in trajectory]
//             trajectory_str = "--trajectory="
//             for i in generator:
//                 trajectory_str += i + ","
//             trajectory_str = trajectory_str[:-1]
//             controller_arguments.insertMFString(0, trajectory_str)
//             controller_arguments.insertMFString(1, "--speed=" + str(speed))

//     def stop(self):
//         controller_arguments = self.handle.getField("controllerArgs")
//         controller_arguments.removeMF(0)
//         controller_arguments.removeMF(1)

//     def get_handle(self):
//         return self.handle

//     def get_velocity(self):
//         return self.handle.getVelocity()

//     def remove(self):
//         self.handle.remove()


#include <fstream>
#include <iostream>
#include <boost/algorithm/string.hpp>

#include <WebotsAPI.h>
#include <Human.h>

Human::Human(const std::string &name, const double &x, const double &z, const double &angle, const std::string &command, WebotsAPI *wapi): WorldEntity(wapi)
{
    this->x = x;
    this->z = z;
    this->angle = angle;
    this->dynamic = false;

    std::string base_str;
    const double rotV[4] = {0., 1., 0., angle};

    base_str = "../../protos/pedestrian_base.wbo";


    std::ifstream ifs(base_str);
    std::string content( (std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()) );
    boost::replace_all(content, "SEEDNAME", name);
    boost::replace_all(content, "SEEDX", std::to_string(x));
    // boost::replace_all(content, "SEEDY", std::to_string(y));
    boost::replace_all(content, "SEEDZ", std::to_string(z));
    boost::replace_all(content, "ORIENT", std::to_string(angle));
    // string = string.replace("ORIENT", str(orient))

    wapi->rootChildren->importMFNodeFromString(-1, content);
    this->handle = this->wapi->getFromDef(name);
    // const double posV[3] = {x, y, z};
    // this->handle->getField("translation")->setSFVec3f(posV);
    
    this->handle->getField("rotation")->setSFRotation(rotV);
    // Node *wallShapeNode = this->wapi->getFromDef(name + "_SHAPE");
    // wallShapeNode->getField("geometry")->getSFNode()->getField("size")->setSFVec3f(&sizeV[0]);

}


bool Human::check_collision()
{
    return false;
}

// void Human::update_position(const double &x, const double &z)
// {
//     const double posV[3] = {x, 1.27, z};
//     this->handle->getField("translation")->setSFVec3f(posV);
// }



// class Object(Entity):
//     def __init__(self, name, otype, x, z, angle, supervisor):
//         super(Object, self).__init__()
//         self.supervisor = supervisor
//         self.rootChildren = supervisor.getRoot().getField("children")
//         if otype == "desk":
//             base_str = "".join(open("../../protos/desk_base.wbo", "r").readlines())
//             self.size_x = DESK_SIZE_X
//             self.size_y = DESK_SIZE_Y
//             self.size_z = DESK_SIZE_Z
//         elif otype == "plant":
//             base_str = "".join(open("../../protos/plant_base.wbo", "r").readlines())
//             self.size_x = PLANT_SIZE_X
//             self.size_y = PLANT_SIZE_Y
//             self.size_z = PLANT_SIZE_Z
//         else:
//             print("Not valid object type")
//             base_str = None

//         string = base_str.replace("SEEDNAME", name)
//         string = string.replace("SEEDX", str(x))
//         string = string.replace("SEEDZ", str(z))
//         self.rootChildren.importMFNodeFromString(-3, string)
//         obj = self.supervisor.getFromDef(name)
//         obj.getField("rotation").setSFRotation([0, 1, 0, angle])

//         self.handle = obj

//     def set_id(self, id):
//         self.id = id

//     def get_id(self):
//         return self.id

//     def get_handle(self):
//         return self.handle

//     def remove(self):
//         self.handle.remove()

//     def get_velocity(self):
//         return self.handle.getVelocity()

//     def check_collision(self):
//         return True if self.handle.getField("boundingObject") is not None else False

//     def get_model_bounding_box(self):
//         x, y = self.get_position()
//         return [
//             x + self.size_x / 2,
//             x - self.size_x / 2,
//             y + self.size_y / 2,
//             y - self.size_y / 2,
//         ]