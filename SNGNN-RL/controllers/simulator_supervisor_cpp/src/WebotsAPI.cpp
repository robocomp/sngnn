#include <WebotsAPI.h>

#include <Goal.h>
#include <RobotNode.h>
#include <Obstacle.h>
#include <Human.h>
#include <Relation.h>

WebotsAPI::WebotsAPI() : Supervisor()
{
    root = getRoot();
    rootChildren = root->getField("children");
}


void WebotsAPI::close()
{
    this->simulationQuit(0);
}


void WebotsAPI::remove_objects()
{
    // printf("%s: %d\n", __FILE__, __LINE__);
    while (rootChildren->getMFNode(5) != rootChildren->getMFNode(-1))
    {
        // printf("%s: %d\n", __FILE__, __LINE__);
        rootChildren->getMFNode(-1)->remove();
    }
}


Wall *WebotsAPI::create_wall(const std::string &name, const double &x1, const double &z1, const double &x2, const double &z2)
{
    return new Wall(name, x1, z1, x2, z2, this);
}


Node *WebotsAPI::create_floor(const std::string &name, double x, double z, double width, double length)
{
    double position[3] = { x, 0.0, z };
    double size[2] = { width, length };
    double tileSize[2] = { 0.5, 0.5 };

    getRoot()->getField("children")->importMFNodeFromString(-1, "DEF " + name + " Floor {}");
    auto floorNode = this->getFromDef(name);
    floorNode->getField("size")->setSFVec2f(size);
    floorNode->getField("translation")->setSFVec3f(position);
    floorNode->getField("tileSize")->setSFVec2f(tileSize);
    floorNode->getField("name")->setSFString(name);
    floorNode->getField("appearance")->getSFNode()->getField("type")->setSFString("chequered");
    // floorNode->getField("appearance")->getSFNode()->remove();

    return floorNode;
}


Goal *WebotsAPI::create_goal(const double &x, const double &z)
{
    return new Goal(x, z, this);
}

RobotNode *WebotsAPI::create_robot(const double &x, const double &z, const double &angle)
{
    return new RobotNode(x, z, angle, this);
}

Obstacle *WebotsAPI::create_object(const std::string &name, const double &x, const double &z, const double &angle, const std::string &otype)
{
    return new Obstacle(name, x, z, angle, otype, this);
}

Human *WebotsAPI::create_humans(const std::string &name, const double &x, const double &z, const double &angle, const std::string &command)
{
    return new Human(name, x, z, angle, command, this);
}

Relation *WebotsAPI::create_relations(const std::string &name, const std::string &relation_type, WorldEntity *node1, WorldEntity *node2)
{
    return new Relation(name, relation_type, node1, node2, this);
}
