#pragma once

#include <WorldEntity.h>


class Relation: public WorldEntity
{
public:

    Relation(const std::string &name, const std::string &relation_type, WorldEntity *node1, WorldEntity *node2, WebotsAPI *wapi);

};
