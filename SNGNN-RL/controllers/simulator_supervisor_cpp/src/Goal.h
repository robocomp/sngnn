#pragma once

#include <WorldEntity.h>

class Goal: public WorldEntity
{
public:

    Goal(const double &x, const double &z, WebotsAPI *wapi);

    bool check_collision();


};



