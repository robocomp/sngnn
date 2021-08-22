#include <fstream>
#include <iostream>
#include <boost/algorithm/string.hpp>

#include <WebotsAPI.h>
#include <Goal.h>

Goal::Goal(const double &x, const double &z, WebotsAPI *wapi): WorldEntity(wapi)
{
    this->x = x;
    this->z = z;
    this->dynamic = true;

    std::ifstream ifs("../../protos/goal_base2.wbo");
    std::string content( (std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()) );
    // boost::replace_all(content, "SEEDNAME", this->name);
    boost::replace_all(content, "SEEDX", std::to_string(x));
    boost::replace_all(content, "SEEDZ", std::to_string(z));

    this->wapi->rootChildren->importMFNodeFromString(-1, content);
    this->handle = this->wapi->getFromDef("GOAL");
}


bool Goal::check_collision()
{
    return false;
}
