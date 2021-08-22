#include <fstream>
#include <iostream>
#include <boost/algorithm/string.hpp>

#include <WebotsAPI.h>
#include <Obstacle.h>

Obstacle::Obstacle(const std::string &name, const double &x, const double &z, const double &angle, const std::string &otype, WebotsAPI *wapi): WorldEntity(wapi)
{
    this->x = x;
    this->z = z;
    this->angle = angle;
    this->dynamic = false;

    std::string base_str;
    std::vector<double> rotV = {0., 1., 0., angle};
    std::vector<double> sizeV = {0., 0., 0.};

    if (otype == "desk")
    {
        base_str = "../../protos/desk_base.wbo";
        sizeV[0] = DESK_SIZE_X;
        sizeV[1] = DESK_SIZE_Y;
        sizeV[2] = DESK_SIZE_Z;
    }
    else if (otype == "plant")
    {
        base_str = "../../protos/plant_base.wbo";
        sizeV[0] = PLANT_SIZE_X;
        sizeV[1] = PLANT_SIZE_Y;
        sizeV[2] = PLANT_SIZE_Z;
    }
    else
    {
        printf("Not valid object type: %s\n", otype.c_str());
        exit(-1);
    }


    std::ifstream ifs(base_str);
    std::string content( (std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()) );
    boost::replace_all(content, "SEEDNAME", name);
    boost::replace_all(content, "SEEDX", std::to_string(x));
    // boost::replace_all(content, "SEEDY", std::to_string(y));
    boost::replace_all(content, "SEEDZ", std::to_string(z));

    wapi->rootChildren->importMFNodeFromString(-1, content);
    this->handle = this->wapi->getFromDef(name);
    // const double posV[3] = {x, y, z};
    // this->handle->getField("translation")->setSFVec3f(posV);
    
    this->handle->getField("rotation")->setSFRotation(&rotV[0]);
    // Node *wallShapeNode = this->wapi->getFromDef(name + "_SHAPE");
    // wallShapeNode->getField("geometry")->getSFNode()->getField("size")->setSFVec3f(&sizeV[0]);

}


bool Obstacle::check_collision()
{
    return false;
}
