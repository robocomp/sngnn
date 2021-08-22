#pragma once

#include <WorldEntity.h>


class Human: public WorldEntity
{
public:

    Human(const std::string &name, const double &x, const double &z, const double &angle, const std::string &command, WebotsAPI *wapi);
    bool check_collision();

};
