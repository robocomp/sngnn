#include <fstream>
#include <iostream>
#include <boost/algorithm/string.hpp>

#include <WebotsAPI.h>
#include <Wall.h>

Wall::Wall(const std::string &name, const double &x1, const double &z1, const double &x2, const double &z2, WebotsAPI *wapi): WorldEntity(wapi)
{
    const double x = 0.5 * (x1 + x2);
    const double z = 0.5 * (z1 + z2);
    const double ix = x2 - x1;
    const double iz = z2 - z1;
    double angle = atan2(iz, ix);

    this->length = sqrt(ix*ix + iz*iz);
    this->name = name;

    this->dynamic = false;

    const double sx = this->length;  // WALL_LENGTH (meters)
    const double sy = 0.3;  // WALL_HEIGHT (meters)
    const double sz = 0.02;  // WALLS_WIDTH (meters)
    const double sizeV[3] = {sx, sy, sz};
    const double y = sy / 2;

    std::ifstream ifs("../../protos/wall_base.wbo");
    std::string content( (std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()) );
    boost::replace_all(content, "SEEDNAME", this->name);
    boost::replace_all(content, "SEEDX", std::to_string(sx));
    boost::replace_all(content, "SEEDY", std::to_string(sy));
    boost::replace_all(content, "SEEDZ", std::to_string(sz));

    this->wapi->rootChildren->importMFNodeFromString(-1, content);
    this->handle = this->wapi->getFromDef(this->name);
    if (this->handle == NULL)
    {
        printf("WALL HANDLE IS NULL\n");
        fflush(stderr);
        exit(1);
    }
    const double posV[3] = {x, y, z};
    this->handle->getField("translation")->setSFVec3f(posV);
    const double rotV[4] = {0, 1, 0, angle};
    this->handle->getField("rotation")->setSFRotation(rotV);
    Node *wallShapeNode = this->wapi->getFromDef(this->name + "_SHAPE");
    wallShapeNode->getField("geometry")->getSFNode()->getField("size")->setSFVec3f(sizeV);
}



double Wall::get_length()
{
    return this->length;
}

