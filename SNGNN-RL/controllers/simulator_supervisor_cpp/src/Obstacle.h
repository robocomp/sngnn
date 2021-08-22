#pragma once

#include <WorldEntity.h>


#define DESK_SIZE_X (1.2)
#define DESK_SIZE_Y (0.02 + 0.66 + 0.357)
#define DESK_SIZE_Z (0.7)
#define PLANT_SIZE_X (0.3)
#define PLANT_SIZE_Y (0.27 + 0.46 + 0.6)
#define PLANT_SIZE_Z (0.3)


class Obstacle: public WorldEntity
{
public:

    Obstacle(const std::string &name, const double &x, const double &z, const double &angle, const std::string &otype, WebotsAPI *wapi);

    bool check_collision();

    std::vector<double> sizes;

};



//         super(Object, self).__init__()
//         this->supervisor = supervisor
//         this->rootChildren = wapi->getRoot()->getField("children")
//         if otype == "desk":
//             base_str = "".join(open("../../protos/desk_base.wbo", "r").readlines())
//             this->size_x = DESK_SIZE_X
//             this->size_y = DESK_SIZE_Y
//             this->size_z = DESK_SIZE_Z
//         elif otype == "plant":
//             base_str = "".join(open("../../protos/plant_base.wbo", "r").readlines())
//             this->size_x = PLANT_SIZE_X
//             this->size_y = PLANT_SIZE_Y
//             this->size_z = PLANT_SIZE_Z
//         else:
//             print("Not valid object type")
//             base_str = None

//         string = base_str.replace("SEEDNAME", name)
//         string = string.replace("SEEDX", str(x))
//         string = string.replace("SEEDZ", str(z))
//         this->rootchildren->importMFNodeFromString(-1, string)
//         obj = this->wapi->getFromDef(name)
//         obj->getField("rotation")->setSFRotation([0, 1, 0, angle])

//         this->handle = obj

//     def set_id(self, id):
//         this->id = id

//     def get_id(self):
//         return this->id

//     def get_handle(self):
//         return this->handle

//     def remove(self):
//         this->handle.remove()

//     def get_velocity(self):
//         return this->handle.getVelocity()

//     def check_collision(self):
//         return True if this->handle->getField("boundingObject") is not None else False

//     def get_model_bounding_box(self):
//         x, y = this->get_position()
//         return [
//             x + this->size_x / 2,
//             x - this->size_x / 2,
//             y + this->size_y / 2,
//             y - this->size_y / 2,
//         ]

