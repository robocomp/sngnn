#pragma once

#include <WorldEntity.h>

class Wall: public WorldEntity
{
public:

    Wall(const std::string &name, const double &x1, const double &z1, const double &x2, const double &z2, WebotsAPI *wapi);


    double get_length();

    // def check_collision(self):
        // return True if this->handle->getField("boundingObject") is not None else False

    // def get_model_bounding_box(self):
        // wallShapeNode = this->wapi->getFromDef(this->name + "_SHAPE")
        // return wallShapeNode->getField("geometry")->getSFNode()->getField("size")

// protected:
    double length;
    std::string name;
};
