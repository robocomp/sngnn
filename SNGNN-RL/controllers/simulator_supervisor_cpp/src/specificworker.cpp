/*
 *    Copyright (C) 2021 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "specificworker.h"
#include <bits/stdc++.h>
#include <iostream>
#include <string>
#include <string.h>
#include <cstdlib>
#include <math.h>
#include <QTime>

using namespace std;


/**
* \brief Default constructor
*/
SpecificWorker::SpecificWorker(MapPrx& mprx) : GenericWorker(mprx)
{
    this->RL_MODE = true;
    this->goalNode = NULL;
    this->robotNode = NULL;

    hide();

    wapi = new WebotsAPI();

    //default length
    this->wall_length = 4.;
}

SpecificWorker::~SpecificWorker()
{
    delete this->wapi;
    if (this->goalNode) delete this->goalNode;
    if (this->robotNode) delete this->robotNode;
}


void SpecificWorker::initialize(int period)
{
	this->Period = period;
	timer.start(Period);
}

void SpecificWorker::compute()
{
    // this->create_scene();
    if (this->RL_MODE)
    {
        while (true)
        {
            usleep(100000);
        }
    }
    else
    {
        printf("No RL mode?");
        fflush(stdout);
        exit(-1);
        int step_time = wapi->getBasicTimeStep();
        while (true)
        {
            this->Simulator_step();
            this->publish_data();
            usleep(1000*step_time);
        }
    }	
}


void SpecificWorker::publish_data()
{
    int lastID = 0;

    // this->bytesequencepublisher_pubproxy->newsequence(...)
	// std::cout << "publish_data" << std::endl;

    RoboCompGoalPublisher::GoalT goal;
    std::vector<RoboCompInteractionDetector::InteractionT> interactions;
    std::vector<RoboCompObjectDetector::ObjectT> objects;
    std::vector<RoboCompPeopleDetector::Person> people;
    std::vector<RoboCompWallDetector::WallT> walls;

    // struct timeval tp;
    // gettimeofday(&tp, NULL);
    // double timestamp = tp.tv_sec + tp.tv_usec / 1000000;
    double timestamp = this->wapi->getTime();

    //printf("%s: %d\n", __FILE__, __LINE__);
    if (this->robotNode)
    {
        RoboCompObjectDetector::ObjectT object_;
        object_.timestamp = timestamp;
        object_.id = -2;
        object_.x = 0;
        object_.y = 0;
        object_.angle = 0;
        object_.ix = 0;
        object_.iy = this->robotNode->actions[0];
        object_.iangle = this->robotNode->actions[2];
        object_.bbx1 = -0.25;
        object_.bbx2 = +0.25;
        object_.bby1 = -0.25;
        object_.bby2 = +0.25;
        object_.collision = this->robotNode->check_collision();
        objects.push_back(object_);
    }
    
    if (this->goalNode)
    {
        goal.timestamp = timestamp;
        goal.x = -this->coordinates[this->goalNode][0];
        goal.y = this->coordinates[this->goalNode][1];
        // printf("  PD (%f, %f)\n", goal.x, goal.y);
    }

    if (this->goalNode==NULL or this->robotNode==NULL)
    {
        goal.x = objects[0].x = 0;
        goal.y = objects[0].y = 0;
    }

    //printf("%s: %d\n", __FILE__, __LINE__);

    for (auto plant : this->plantNodes)
    {
        RoboCompObjectDetector::ObjectT object_;
        object_.timestamp = timestamp;
        object_.id = ++lastID;

        object_.x = -this->coordinates[plant][0];
        object_.y = this->coordinates[plant][1];
        object_.angle = -M_PIl/2.-this->coordinates[plant][2];//+M_PIl/2.;
        object_.ix = object_.x - (-this->prev_coordinates[plant][0]);
        object_.iy = object_.y - (this->prev_coordinates[plant][1]);
        const double ia = object_.angle - this->prev_coordinates[plant][2];
        object_.iangle = atan2(sin(ia), cos(ia));
        object_.bbx1 = -0.15;
        object_.bbx2 = +0.15;
        object_.bby1 = -0.15;
        object_.bby2 = +0.15;
        object_.collision = false;
        objects.push_back(object_);
    }

    // printf("%s: %d\n", __FILE__, __LINE__);

    for (auto desk : this->deskNodes)
    {
        RoboCompObjectDetector::ObjectT object_;
        object_.timestamp = timestamp;
        object_.id = ++lastID;

        object_.x = -this->coordinates[desk][0];
        object_.y = this->coordinates[desk][1];
        object_.angle = -M_PIl/2.-this->coordinates[desk][2];//+M_PIl/2.;
        object_.ix = object_.x - (-this->prev_coordinates[desk][0]);
        object_.iy = object_.y - (this->prev_coordinates[desk][1]);
        const double ia = object_.angle - this->prev_coordinates[desk][2];
        object_.iangle = atan2(sin(ia), cos(ia));
        object_.bbx1 = -0.15;
        object_.bbx2 = +0.15;
        object_.bby1 = -0.15;
        object_.bby2 = +0.15;
        object_.collision = false;
        objects.push_back(object_);
    }

    for (auto wall : this->wallNodes)
    {
        RoboCompWallDetector::WallT w;
        w.timestamp = timestamp;
        w.id = ++lastID;
        const double wl = wall->get_length();
        w.x1 = -(this->coordinates[wall][0] + cos(this->coordinates[wall][2]) * wl / 2.0);
        w.y1 = (this->coordinates[wall][1] - sin(this->coordinates[wall][2]) * wl / 2.0);
        w.x2 = -(this->coordinates[wall][0] - cos(this->coordinates[wall][2]) * wl / 2.0);
        w.y2 = (this->coordinates[wall][1] + sin(this->coordinates[wall][2]) * wl / 2.0);
        walls.push_back(w);
    }

    //printf("%s: %d\n", __FILE__, __LINE__);


    for (auto human : this->humanNodes)
    {
        RoboCompPeopleDetector::Person per;
        per.timestamp = timestamp;
        per.id = ++lastID;

        per.x = -this->coordinates[human][0];
        per.y = this->coordinates[human][1];
        per.angle = -M_PIl/2.-this->coordinates[human][2];//+M_PIl/2.;
        per.ix = per.x - (-this->prev_coordinates[human][0]);
        per.iy = per.y - (this->prev_coordinates[human][1]);
        const double ia = per.angle - this->prev_coordinates[human][2];
        per.iangle = atan2(sin(ia), cos(ia));
        people.push_back(per);
    }
    
    // for human in enumerate(this->peopleNodes):
    //     person = Person()
    //     person.timestamp = timestamp
    //     person.id = ++lastID;
    //     human.set_id(person.id)
    //     person.x, person.y = human.get_position(relative_to=this->robotNode)
    //     person.angle = human.get_orientation(relative_to=this->robotNode)
    //     person.ix, _, person.iy, _, person.iangle, _ = human.get_velocity()
    //     people.push_back(person)

    // for counter, desk in enumerate(this->deskNodes):
    //        ObjectT object_;
    //     object_.timestamp = timestamp
    //     object_.id = ++lastID;
    //     object_.x, object_.y = desk.get_position(relative_to=this->robotNode)
    //     object_.angle = desk.get_orientation(relative_to=this->robotNode)
    //     object_.ix, _, object_.iy, _, object_.iangle, _ = desk.get_velocity()
    //     bounding_box = desk.get_model_bounding_box()
    //     object_.bbx1 = bounding_box[0]
    //     object_.bby1 = bounding_box[2]
    //     object_.bbx2 = bounding_box[1]
    //     object_.bby2 = bounding_box[3]
    //     objects.push_back(object_)


    // for interaction in this->interactions:
    //     i = InteractionT();
    //     i.timestamp = timestamp;
    //     i.idSrc = interaction.node1.get_id();  # inter["src"] - len(this->walls_IND)
    //     i.idDst = interaction.node2.get_id();  # inter["dst"] - len(this->walls_IND)
    //     i.type = interaction.relation_type;  # inter["relationship"]
    //     interactions.push_back(i);



    // printf("%s: %d\n", __FILE__, __LINE__);
    this->peopledetector_pubproxy->gotpeople(people);
    // printf("%s: %d\n", __FILE__, __LINE__);
    this->objectdetector_pubproxy->gotobjects(objects);
    //printf("%s: %d\n", __FILE__, __LINE__);
    this->interactiondetector_pubproxy->gotinteractions(interactions);
    //printf("%s: %d\n", __FILE__, __LINE__);
    this->walldetector_pubproxy->gotwalls(walls);
    //printf("%s: %d\n", __FILE__, __LINE__);
    this->goalpublisher_pubproxy->goalupdated(goal);
    // printf("%s: %d\n", __FILE__, __LINE__);
}


void SpecificWorker::OmniRobot_correctOdometer(const int x, const int z, const float alpha)
{
//implementCODE

}

void SpecificWorker::OmniRobot_getBasePose(int &x, int &z, float &alpha)
{
//implementCODE

}

void SpecificWorker::OmniRobot_getBaseState(RoboCompGenericBase::TBaseState &state)
{
//implementCODE

}

void SpecificWorker::OmniRobot_resetOdometer()
{
//implementCODE

}

void SpecificWorker::OmniRobot_setOdometer(const RoboCompGenericBase::TBaseState &state)
{
//implementCODE

}

void SpecificWorker::OmniRobot_setOdometerPose(const int x, const int z, const float alpha)
{
//implementCODE

}

void SpecificWorker::OmniRobot_setSpeedBase(const float advx, const float advz, const float rot)
{
    double actions[4];
    actions[0] = advx;
    actions[1] = advz;
    actions[2] = rot;
    if (this->robotNode)
    {
        this->robotNode->set_actions(actions);
    }
}

void SpecificWorker::OmniRobot_stopBase()
{
    OmniRobot_setSpeedBase(0.0, 0.0, 0.0);
}

void SpecificWorker::Simulator_regenerate(const std::string &scene)
{
    this->wapi->remove_objects();

    this->clear_entities_list();

    this->create_scene(scene);

    this->initialise_coordinates_map();
}

void SpecificWorker::clear_entities_list()
{
    for (auto n : this->wallNodes)
        delete n;
    for (auto n : this->plantNodes)
        delete n;
    for (auto n : this->deskNodes)
        delete n;
    for (auto n : this->humanNodes)
        delete n;
    for (auto n : this->relationNodes)
        delete n;

    this->wallNodes.clear();
    this->plantNodes.clear();
    this->deskNodes.clear();
    this->humanNodes.clear();
    this->floorNodes.clear();
    this->relationNodes.clear();
    this->human_goals.clear(); 
}

    // std::unordered_map<void *, std::vector<float>> coordinates, prev_coordinates;
void SpecificWorker::initialise_coordinates_map()
{
    this->coordinates.clear();
    this->compute_current_coordinates();
    this->prev_coordinates = this->coordinates;   
}

void SpecificWorker::compute_current_coordinates()
{
    std::vector<double> v3 = {0., 0., 0.};

    if (this->goalNode==NULL or this->robotNode==NULL)
    {
        return;
    }

    // Robot node is always 0s.
    coordinates[this->robotNode] = v3;
    this->robotNode->get_position(v3[0], v3[1]);
    this->goalNode->get_orientation(v3[2]);

    // Goal, just one
    // printf("------------------------------------------------------------------\n");
    this->goalNode->get_position(v3[0], v3[1]);
    // printf(" CCG (%f, %f)\n", v3[0], v3[1]);

    this->goalNode->get_orientation(v3[2], this->robotNode);
    this->goalNode->get_position(v3[0], v3[1], this->robotNode);
    // printf(" CCR (%f, %f)\n", v3[0], v3[1]);
    coordinates[this->goalNode] = v3;
    // printf("------------------------------------------------------------------\n");

    // Plants
    for (auto plant : this->plantNodes)
    {
        plant->get_orientation(v3[2], this->robotNode);
        plant->get_position(v3[0], v3[1], this->robotNode);
        coordinates[plant] = v3;
    }

    // Desks
    for (auto desk : this->deskNodes)
    {
        desk->get_orientation(v3[2], this->robotNode);
        desk->get_position(v3[0], v3[1], this->robotNode);
        coordinates[desk] = v3;
    }

    // Humans
    for (auto human : this->humanNodes)
    {
        human->get_orientation(v3[2], this->robotNode);
        human->get_position(v3[0], v3[1], this->robotNode);
        coordinates[human] = v3;
    }

    // Walls
    for (auto wall : this->wallNodes)
    {
        wall->get_position(v3[0], v3[1], this->robotNode);
        wall->get_orientation(v3[2], this->robotNode);
        coordinates[wall] = v3;
    }

}

void SpecificWorker::clear_dynamic_checks()
{
    for (auto n : this->wallNodes)
        n->clear_dynamic_check();

    for (auto n : this->plantNodes)
        n->clear_dynamic_check();

    for (auto n : this->deskNodes)
        n->clear_dynamic_check();

    for (auto n : this->humanNodes)
        n->clear_dynamic_check();

    if (this->robotNode)
    {
        this->robotNode->clear_dynamic_check();
    }
    if (this->goalNode)
    {
        this->goalNode->clear_dynamic_check();
    }
}

void SpecificWorker::Simulator_step(const int timestep)
{
    // MAKE SURE NOT TO MOVE STATIONARY HUMANS
    if (this->HUMAN_INIT_MODE != 0)
    {
        double vel = 0.015; //get_random_double(0.01, 0.08);
        double angle, ix, iz, i;
        for (auto human : this->humanNodes)
        {
            if (sqrt(pow(this->human_goals[i][0]-human->x,2)+pow(this->human_goals[i][1]-human->z,2))>0.1)
            {
                ix = this->human_goals[i][0]-human->x;
                iz = this->human_goals[i][1]-human->z;
                angle = atan2(iz, ix);
                human->set_world_position(human->x + vel*cos(angle), human->z + vel*sin(angle));
            }
            i++;
        }
    }

    if (timestep>0)
    {
        this->wapi->step(timestep);
    }
    else
    {
        this->clear_dynamic_checks();        
        QTime a = QTime::currentTime();
        int ttt = wapi->getBasicTimeStep();
        this->wapi->step(ttt);
        a = QTime::currentTime();
    }

    if (this->RL_MODE)
    {
        //printf("%s: %d\n", __FILE__, __LINE__);
        this->compute_current_coordinates();
        //printf("%s: %d\n", __FILE__, __LINE__);
        this->publish_data();
        //printf("%s: %d\n", __FILE__, __LINE__);
    }
    //printf("%s: %d\n", __FILE__, __LINE__);
}

void SpecificWorker::create_scene(const std::string &scene)
{

    std::vector<std::string> params;
    std::string p;
    std::istringstream paramsStream(scene);
    while (std::getline(paramsStream, p, ' '))
        params.push_back(p);

    int number_of_static_humans = stoi(params[0]);
    int number_of_dynamic_humans = stoi(params[1]);
    int number_of_plants = stoi(params[2]);
    int number_of_tables = stoi(params[3]);
    int number_of_relations = stoi(params[4]);
    int include_walls = stoi(params[5]);

    // printf("SCENE \n");
    // cout << scene << endl;
    // cout << number_of_static_humans << number_of_dynamic_humans << number_of_plants << number_of_tables << number_of_relations  << endl;

    //number_of_dynamic_humans = 0;
    number_of_plants = 0;
    number_of_tables = 0;
    number_of_relations = 0;

    // Build the room
    this->create_walls_and_floor(include_walls);
    // Create robot and goal nodes
    this->create_robot_and_goal();
    // Create plants
    this->create_plants(number_of_plants);

    // # Create humans 0-->random pose but move; 1--> square and move to the other side; 2--> circle move diametrically opp; 3--> random pose; 4--> test bed (square/circle/random with varied number of humans)
    this->HUMAN_INIT_MODE = 2;//rand()%4;

    if (this->HUMAN_INIT_MODE != 4)
    {
        number_of_static_humans = rand()%8 + 1;

        this->create_humans(number_of_static_humans);

        // # Create desk
        this->create_desks(number_of_tables);
        // # Create relations
        this->create_relations(number_of_relations);
    }
    else
    {
        this->type = "random"; // circle, square, random, stationary
        this->level = 5; // 1 --> 1 Human; 2 --> 2 Humans; 3 --> 4 Humans; 4 --> 6 Humans; 5 --> 8 Humans
        this->create_test_bed();
    }
    
}


void SpecificWorker::create_walls_and_floor(int walls)
{
    if (walls)
    {
        // SQUARE
        // if (this->wall_type == 0)
        {
            this->wall_length = 4.; // random.randint(6, 12)

            this->wallNodes.push_back(
                this->wapi->create_wall("W1",
                    -this->wall_length / 2, +this->wall_length / 2,
                    -this->wall_length / 2, -this->wall_length / 2));

            // The endpoints of horizontal walls have to be exchanged
            this->wallNodes.push_back(
                this->wapi->create_wall("W2",
                    this->wall_length / 2, -this->wall_length / 2,
                    -this->wall_length / 2, -this->wall_length / 2));


            this->wallNodes.push_back(
                this->wapi->create_wall("W3",
                    this->wall_length / 2, -this->wall_length / 2,
                    this->wall_length / 2, this->wall_length / 2));

            // The endpoints of horizontal walls have to be exchanged
            this->wallNodes.push_back(
                this->wapi->create_wall("W4",
                    -this->wall_length / 2, +this->wall_length / 2,
                    this->wall_length / 2, +this->wall_length / 2));

            this->floorNodes.push_back(
                this->wapi->create_floor("F1", 0., 0., this->wall_length, this->wall_length)
            );
        }


        // # RECTANGLE
        // elif this->wall_type == 1:
        //     this->wall_breadth = random.randint(10, 15)
        //     this->wall_length = random.randint(this->wall_breadth, 20)

        //     p1 = [this->wall_length / 2, this->wall_breadth / 2]
        //     p2 = [this->wall_length / 2, -this->wall_breadth / 2]
        //     this->wallNodes.push_back(this->create_wall("W1", p1, p2))

        //     p1 = [this->wall_length / 2, -this->wall_breadth / 2]
        //     p2 = [-this->wall_length / 2, -this->wall_breadth / 2]
        //     this->wallNodes.push_back(this->create_wall("W2", p1, p2))

        //     p1 = [-this->wall_length / 2, -this->wall_breadth / 2]
        //     p2 = [-this->wall_length / 2, this->wall_breadth / 2]
        //     this->wallNodes.push_back(this->create_wall("W3", p1, p2))

        //     p1 = [-this->wall_length / 2, this->wall_breadth / 2]
        //     p2 = [this->wall_length / 2, this->wall_breadth / 2]
        //     this->wallNodes.push_back(this->create_wall("W4", p1, p2))

        //     this->floorNodes.push_back(
        //         this->create_floor("F1", [this->wall_length, this->wall_breadth], [0, 0])
        //     )

        // # L Shaped
        // elif this->wall_type == 2:
        //     this->wall_length = random.randint(7, 10)

        //     p1 = [this->wall_length / 2, this->wall_length]
        //     p2 = [this->wall_length / 2, -this->wall_length]
        //     this->wallNodes.push_back(this->create_wall("W1", p1, p2))

        //     p1 = [this->wall_length / 2, -this->wall_length]
        //     p2 = [-1.5 * this->wall_length, -this->wall_length]
        //     this->wallNodes.push_back(this->create_wall("W2", p1, p2))

        //     p1 = [-1.5 * this->wall_length, -this->wall_length]
        //     p2 = [-1.5 * this->wall_length, 0]
        //     this->wallNodes.push_back(this->create_wall("W3", p1, p2))

        //     p1 = [-1.5 * this->wall_length, 0]
        //     p2 = [-this->wall_length / 2, 0]
        //     this->wallNodes.push_back(this->create_wall("W4", p1, p2))

        //     p1 = [-this->wall_length / 2, 0]
        //     p2 = [-this->wall_length / 2, this->wall_length]
        //     this->wallNodes.push_back(this->create_wall("W5", p1, p2))

        //     p1 = [-this->wall_length / 2, this->wall_length]
        //     p2 = [this->wall_length / 2, this->wall_length]
        //     this->wallNodes.push_back(this->create_wall("W6", p1, p2))

        //     this->floorNodes.push_back(
        //         this->create_floor(
        //             "F1",
        //             [2 * this->wall_length, this->wall_length],
        //             [-this->wall_length / 2, -this->wall_length / 2],
        //         )
        //     )
        //     this->floorNodes.push_back(
        //         this->create_floor(
        //             "F2",
        //             [this->wall_length, this->wall_length],
        //             [0, this->wall_length / 2],
        //         )
        //     )
    }
    else
    {
        this->floorNodes.push_back(this->wapi->create_floor("F1", 0., 0., 20, 20));
    }
}


void SpecificWorker::create_robot_and_goal()
{
    double orient = get_random_double(0., 2.*M_PIl);
    // Create robot
    this->place_the_object(robot_p[0], robot_p[1], 0.5);
    if (this->robotNode) delete this->robotNode;
    this->robotNode = this->wapi->create_robot(robot_p[0], robot_p[1], orient);

    // Create goal
    double ix, iz;
    std::vector<double> goal_p = {0., 0.};
    do
    {
        this->place_the_object(goal_p[0], goal_p[1], 0.3);
        ix = (robot_p[0]-goal_p[0]);
        iz = (robot_p[1]-goal_p[1]);
    } while (sqrt(ix*ix + iz*iz) < 0.7);

    if (this->goalNode) delete this->goalNode;
    this->goalNode = this->wapi->create_goal(goal_p[0], goal_p[1]);
}

double get_random_double(const double &min_v, const double &max_v)
{
    const double inc = max_v - min_v;
    return (inc * double(rand()) / double(RAND_MAX)) + min_v;
}


double get_random_int(const int &min_v, const int &max_v)
{
    const double inc = max_v - min_v;
    return int(0.5 + (inc * double(rand()) / double(RAND_MAX)) + min_v);
}

void SpecificWorker::place_human_square(double &x, double &z, const double &threshold, int side, double &angle, double &x_g, double &z_g)
{
    // side = 0 (left), 1 (down), 2 (right), 3 (up)
    if (side == 0)
    {
        x = get_random_double(
        -this->wall_length / 2.0 + threshold, -this->wall_length / 2.0 + 2 * threshold
        );
        z = get_random_double(
        -this->wall_length / 2.0 + threshold, this->wall_length / 2.0 - threshold
        );
        x_g = get_random_double(
        this->wall_length / 2.0 - 2 * threshold, this->wall_length / 2.0 - threshold
        );
        z_g = z;
        // angle = -M_PIl/2.0;
        angle = atan2(x_g-x,z_g-z);
    }
    else if (side == 1)
    {
        z = get_random_double(
        -this->wall_length / 2.0 + threshold, -this->wall_length / 2.0 + 2 * threshold
        );
        x = get_random_double(
        -this->wall_length / 2.0 + threshold, this->wall_length / 2.0 - threshold
        );
        z_g = get_random_double(
        this->wall_length / 2.0 - 2 * threshold, this->wall_length / 2.0 - threshold
        );
        x_g = x;
        // angle = -M_PIl;
        angle = atan2(x_g-x,z_g-z);
    }
    else if (side == 2)
    {
        x = get_random_double(
        this->wall_length / 2.0 - 2 * threshold, this->wall_length / 2.0 - threshold
        );
        z = get_random_double(
        -this->wall_length / 2.0 + threshold, this->wall_length / 2.0 - threshold
        );
        x_g = get_random_double(
        -this->wall_length / 2.0 + threshold, -this->wall_length / 2.0 + 2 * threshold
        );
        z_g = z;
        // angle = M_PIl/2.0;
        angle = atan2(x_g-x,z_g-z);
    }
    else if (side == 3)
    {
        z = get_random_double(
        this->wall_length / 2.0 - 2 * threshold, this->wall_length / 2.0 - threshold
        );
        x = get_random_double(
        -this->wall_length / 2.0 + threshold, this->wall_length / 2.0 - threshold
        );
        z_g = get_random_double(
        -this->wall_length / 2.0 + threshold, -this->wall_length / 2.0 + 2 * threshold
        );
        x_g = x;
        // angle = M_PIl;
        angle = atan2(x_g-x,z_g-z);
    }
}




void SpecificWorker::place_human_circle(double &x, double &z, const double &threshold, double &angle, double &x_g, double &z_g)
{
    // radius of the circle
    double r = get_random_double(
        this->wall_length / 2.0 - 2 * threshold, this->wall_length / 2.0 - threshold
        );
    double angle_ = get_random_double(-M_PIl, M_PIl);
    x = sin(angle_) * -r;
    z = cos(angle_) * -r;
    x_g = sin(angle_) * r;
    z_g = cos(angle_) * r;
    angle = atan2(x_g-x,z_g-z);
}


void SpecificWorker::place_human_random(double &x, double &z, double &angle, double &x_g, double &z_g)
{
    x = get_random_double(
        -this->wall_length / 2.0, this->wall_length / 2.0
        );
    z = get_random_double(
        -this->wall_length / 2.0, this->wall_length / 2.0
        );
    x_g = get_random_double(
        -this->wall_length / 2.0, this->wall_length / 2.0
        );
    z_g = get_random_double(
        -this->wall_length / 2.0, this->wall_length / 2.0
        );
    angle = atan2(x_g-x,z_g-z);
}


void SpecificWorker::place_the_object(double &x, double &z, const double &threshold)
{
    // if this->wall_type == 0:
    x = get_random_double(
        -this->wall_length / 2.0 + threshold, this->wall_length / 2.0 - threshold
    );
    z = get_random_double(
        -this->wall_length / 2.0 + threshold, this->wall_length / 2.0 - threshold
    );
    // elif this->wall_type == 1:
    // x = random.uniform(
    //     -this->wall_length / 2.0 + threshold, this->wall_length / 2.0 - threshold
    // )
    // y = random.uniform(
    //     -this->wall_breadth / 2.0 + threshold,
    //     this->wall_breadth / 2.0 - threshold,
    // )
// elif this->wall_type == 2:
//     x = random.uniform(
//         -1.5 * this->wall_length + threshold, 0.5 * this->wall_length - threshold
//     )
//     if x > -0.5 * this->wall_length + 1:
//         y = random.uniform(
//             -this->wall_length + threshold, this->wall_length - threshold
//         )
//     else:
//         y = random.uniform(-this->wall_length + threshold, 0.0 - threshold)
// return x, y
}

void SpecificWorker::create_plants(const int &n_plants)
{
    std::vector<double> vect = {0., 0., 0.};

    printf("PLANTS: %d\n", n_plants);
    for (int i=0; i<n_plants; i++)
    {
        double angle = get_random_double(-M_PIl, M_PIl);

        double ix, iz;
        std::vector<double> plant_p = {0., 0.};
        int n_tries = 20;
        do
        {
            this->place_the_object(plant_p[0], plant_p[1], 0.3);
            ix = (robot_p[0]-plant_p[0]);
            iz = (robot_p[1]-plant_p[1]);
            n_tries--;
        } while (sqrt(ix*ix + iz*iz) < 0.8 and n_tries>0);

        if(n_tries>0)
            this->plantNodes.push_back(this->wapi->create_object(std::string("plant_")+std::to_string(i), plant_p[0], plant_p[1], angle, "plant"));
    }

}


void SpecificWorker::create_desks(const int &n_desks)
{
    std::vector<double> vect = {0., 0., 0.};

    printf("DESKS: %d\n", n_desks);
    for (int i=0; i<n_desks; i++)
    {
        double angle = get_random_double(-M_PIl, M_PIl);

        double ix, iz;
        std::vector<double> desk_p = {0., 0.};
        int n_tries = 20;
        do
        {
            this->place_the_object(desk_p[0], desk_p[1], 0.8);
            ix = (robot_p[0]-desk_p[0]);
            iz = (robot_p[1]-desk_p[1]);
            n_tries--;
        } while (sqrt(ix*ix + iz*iz) < 0.8 and n_tries>0);

        if(n_tries>0)
            this->deskNodes.push_back(this->wapi->create_object(std::string("desk_")+std::to_string(i), desk_p[0], desk_p[1], angle, "desk"));
    }

}


void SpecificWorker::create_humans(const int &n_humans)
{
    double threshold = 0.5;
    std::vector<double> vect = {0., 0., 0.};
    if (this->HUMAN_INIT_MODE == 1)
    {
        printf("SQUARE FORMATION OF MOVING HUMANS: %d\n", n_humans);
        for (int i=0; i<n_humans; i++)
        {
            int side = i%4;

            double ix, iz, angle, ix_g, iz_g;
            std::vector<double> human_p = {0., 0.};
            std::vector<double> human_goal = {0., 0.};
            int n_tries = 20;
            do
            {
                this->place_human_square(human_p[0], human_p[1], threshold, side, angle, human_goal[0], human_goal[1]);
                ix = (robot_p[0]-human_p[0]);
                iz = (robot_p[1]-human_p[1]);
                ix_g = (robot_p[0]-human_goal[0]);
                iz_g = (robot_p[1]-human_goal[1]);
                n_tries--;
            } while ((sqrt(ix*ix + iz*iz) < 0.8 or sqrt(ix_g*ix_g + iz_g*iz_g) < 0.8) and n_tries>0);

            if(n_tries>0)
            {
                this->human_goals.push_back(human_goal);
                this->humanNodes.push_back(this->wapi->create_humans(std::string("static_human_")+std::to_string(i), human_p[0], human_p[1], angle, "None"));
            }
        }
    }
    else if (this->HUMAN_INIT_MODE == 2)
    {

        printf("CIRCLE FORMATION OF MOVING HUMANS: %d\n", n_humans);
        for (int i=0; i<n_humans; i++)
        {
            double ix, iz, angle, ix_g, iz_g;
            std::vector<double> human_p = {0., 0.};
            std::vector<double> human_goal = {0., 0.};
            int n_tries = 20;
            do
            {
                this->place_human_circle(human_p[0], human_p[1], threshold, angle, human_goal[0], human_goal[1]);
                ix = (robot_p[0]-human_p[0]);
                iz = (robot_p[1]-human_p[1]);
                ix_g = (robot_p[0]-human_goal[0]);
                iz_g = (robot_p[1]-human_goal[1]);
                n_tries--;
            } while ((sqrt(ix*ix + iz*iz) < 1.2 or sqrt(ix_g*ix_g + iz_g*iz_g) < 1.2) and n_tries>0);

            if(n_tries>0)
            {
                this->human_goals.push_back(human_goal);
                this->humanNodes.push_back(this->wapi->create_humans(std::string("static_human_")+std::to_string(i), human_p[0], human_p[1], angle, "None"));
            }
        }
    }
    else if (this->HUMAN_INIT_MODE == 0)
    {

        printf("STATIC HUMANS: %d\n", n_humans);
        for (int i=0; i<n_humans; i++)
        {
            double angle = get_random_double(-M_PIl, M_PIl);

            double ix, iz;
            std::vector<double> human_p = {0., 0.};
            int n_tries = 20;
            do
            {
                this->place_the_object(human_p[0], human_p[1], threshold);
                ix = (robot_p[0]-human_p[0]);
                iz = (robot_p[1]-human_p[1]);
                n_tries--;
            } while (sqrt(ix*ix + iz*iz) < 0.8 and n_tries>0);

            if(n_tries>0)
                this->humanNodes.push_back(this->wapi->create_humans(std::string("static_human_")+std::to_string(i), human_p[0], human_p[1], angle, "None"));
        }
    }
    else if (this->HUMAN_INIT_MODE == 3)
    {

        printf("RANDOM FORMATION OF MOVING HUMANS: %d\n", n_humans);
        for (int i=0; i<n_humans; i++)
        {
            double ix, iz, angle, ix_g, iz_g;
            std::vector<double> human_p = {0., 0.};
            std::vector<double> human_goal = {0., 0.};
            int n_tries = 20;
            do
            {
                this->place_human_random(human_p[0], human_p[1], angle, human_goal[0], human_goal[1]);
                ix = (robot_p[0]-human_p[0]);
                iz = (robot_p[1]-human_p[1]);
                ix_g = (robot_p[0]-human_goal[0]);
                iz_g = (robot_p[1]-human_goal[1]);
                n_tries--;
            } while ((sqrt(ix*ix + iz*iz) < 1.2 or sqrt(ix_g*ix_g + iz_g*iz_g) < 1.2) and n_tries>0);

            if(n_tries>0)
            {
                this->human_goals.push_back(human_goal);
                this->humanNodes.push_back(this->wapi->create_humans(std::string("static_human_")+std::to_string(i), human_p[0], human_p[1], angle, "None"));
            }
        }
    }

}


void SpecificWorker::create_dyn_humans(const int &n_humans)
{
    std::vector<double> vect = {0., 0., 0.};

    printf("DYNAMIC HUMANS: %d\n", n_humans);
    for (int i=0; i<n_humans; i++)
    {
        double angle = get_random_double(-M_PIl, M_PIl);

        double ix, iz;
        std::vector<double> human_p = {0., 0.};
        int n_tries = 20;
        do
        {
            this->place_the_object(human_p[0], human_p[1], 1.2);
            ix = (robot_p[0]-human_p[0]);
            iz = (robot_p[1]-human_p[1]);
            n_tries--;
        } while (sqrt(ix*ix + iz*iz) < 0.8 and n_tries>0);

        // trajectory (last vector is speed)
        vector<vector<double>> trajectory;
        for (int i=0; i<3; i++)
        {
            double ix_, iz_;
            std::vector<double> next_p = {0., 0.};
            int n_tries_ = 20;
            do
            {
                this->place_the_object(next_p[0], next_p[1], 1.2);
                ix_ = (robot_p[0]-next_p[0]);
                iz_ = (robot_p[1]-next_p[1]);
                n_tries_--;
            } while (sqrt(ix_*ix_ + iz_*iz_) < 0.8 and n_tries_>0);
            if(n_tries_>0)
            {
                trajectory.push_back(next_p);
            }
        }
        if (trajectory.size()>0)
        {
            std::vector<double> velocity = {0., 0.};
            // M + (rand() / ( RAND_MAX / (N-M) ) ) ;
            // velocity between 1.0 to 2.5
            velocity[0] = 0.2 + (rand() / ( RAND_MAX / (1.) ) ) ;
            velocity[1] = 0.2 + (rand() / ( RAND_MAX / (1.) ) ) ;
            trajectory.push_back(velocity);
        }
        std::string command;
        for (uint32_t i=0; i<trajectory.size(); i++)
        {
            for (uint32_t j=0; j<trajectory[i].size(); j++)
            {
                command.append(to_string(trajectory[i][j]));
                command.append(" ");
            }
            command.append(",");
        }

        if(n_tries>0)
            this->humanNodes.push_back(this->wapi->create_humans(std::string("dynamic_human_")+std::to_string(i), human_p[0], human_p[1], angle, command));
    }

}


void SpecificWorker::create_relations(const int &n_relations)
{
    std::vector<double> vect = {0., 0., 0.};

    printf("NUMBER OF RELATIONS: %d\n", n_relations);
    for (int i=0; i<n_relations; i++)
    {
        int rel_type = rand()%4;
        rel_type = 0;
        // static_human - static_human: 0; static_human - laptop: 1; static_human - plant: 2; dynamic_human - dynamic_human: 3; 

        if (rel_type == 0)
        {
            int n_tries = 20;
            double angle = get_random_double(0, 2*M_PI);
            // double angle = M_PI;
            double angle_ = angle + M_PI_2*2;
            // double angle_ = M_PI+M_PI;
            // if (angle+M_PI>2*M_PI)
            // {
            //     angle_ = 2*M_PI - (angle+M_PI);
            // }
            // else
            // {
            //     angle_ = angle+M_PI;
            // }
            double ix, iz, ix_, iz_;
            std::vector<double> human_p = {0., 0.};
            
            double dist = 0.8 + (rand()%3)/10.0;
            double delta_x = dist*sin(angle), delta_y = dist*cos(angle);
            std::vector<double> human_p_ = {human_p[0]+delta_x, human_p[1]+delta_y};
            do
            {
                this->place_the_object(human_p[0], human_p[1], 1.2);
                ix = (robot_p[0]-human_p[0]);
                iz = (robot_p[1]-human_p[1]);

                human_p_ = {human_p[0]+delta_x, human_p[1]+delta_y};
                if ((human_p_[0] > -this->wall_length / 2.0 + 1.2) and (human_p_[0] < this->wall_length / 2.0 + 1.2) and (human_p_[1] > -this->wall_length / 2.0 + 1.2) and (human_p_[1] < this->wall_length / 2.0 + 1.2))
                {
                    ix_ = (robot_p[0]-human_p_[0]);
                    iz_ = (robot_p[1]-human_p_[1]);
                }
                else
                {
                    n_tries--;
                    continue;
                }

                n_tries--;
            } while ((sqrt(ix*ix + iz*iz) < 0.8 or sqrt(ix_*ix_ + iz_*iz_) < 0.8) and n_tries>0);

            if(n_tries>0)
            {   
                printf("H1 pose %0.2f %0.2f, orient %0.6f\n", human_p[0], human_p[1], angle);
                printf("H2 pose %0.2f %0.2f, orient %0.6f\n", human_p_[0], human_p_[1], angle_);
                printf("Static Human - Static Human Relaton\n");
                this->humanNodes.push_back(this->wapi->create_humans(std::string("static_human_r_")+std::to_string(i), human_p[0], human_p[1], angle, "None"));
                Human *human2 = this->humanNodes.back();
                this->humanNodes.push_back(this->wapi->create_humans(std::string("static_human_r_")+std::to_string(i+1000), human_p_[0], human_p_[1], angle_, "None"));
                // auto& second_to_last = *std::prev(this->humanNodes.end(), 2);
                this->relationNodes.push_back(this->wapi->create_relations(std::string("relation_")+std::to_string(i), std::string("SH_SH"), this->humanNodes.back(), human2));
            }
        }
        else if (rel_type == 1)
        {
            int n_tries = 20;
            double angle = get_random_double(0, 2*M_PI);
            double angle_ = angle+M_PI_2;
            double ix, iz, ix_, iz_;
            std::vector<double> human_p = {0., 0.};
            
            double dist = 0.8 + (rand()%3)/10.0;
            double delta_x = dist*sin(angle), delta_y = dist*cos(angle);
            std::vector<double> desk_p = {human_p[0]+delta_x, human_p[1]+delta_y};
            do
            {
                this->place_the_object(human_p[0], human_p[1], 1.2);
                ix = (robot_p[0]-human_p[0]);
                iz = (robot_p[1]-human_p[1]);

                desk_p = {human_p[0]+delta_x, human_p[1]+delta_y};
                if ((desk_p[0] > -this->wall_length / 2.0 + 1.2) and (desk_p[0] < this->wall_length / 2.0 + 1.2) and (desk_p[1] > -this->wall_length / 2.0 + 1.2) and (desk_p[1] < this->wall_length / 2.0 + 1.2))
                {
                    ix_ = (robot_p[0]-desk_p[0]);
                    iz_ = (robot_p[1]-desk_p[1]);
                }
                else
                {
                    n_tries--;
                    continue;
                }
                n_tries--;
            } while ((sqrt(ix*ix + iz*iz) < 0.8 or sqrt(ix_*ix_ + iz_*iz_) < 0.8) and n_tries>0);
            printf("Wall length %.6f \n", this->wall_length);
            printf("Human position %.6f %.6f \n", human_p[0], human_p[1]);
            printf("Desk position %.6f %.6f \n", desk_p[0], desk_p[1]);

            if(n_tries>0)
            {
                printf("Static Human - Desk Relaton\n");
                this->humanNodes.push_back(this->wapi->create_humans(std::string("static_human_r_")+std::to_string(i), human_p[0], human_p[1], angle, "None"));
                this->deskNodes.push_back(this->wapi->create_object(std::string("desk_r_")+std::to_string(i), desk_p[0], desk_p[1], angle_, "desk"));
                this->relationNodes.push_back(this->wapi->create_relations(std::string("relation_")+std::to_string(i), std::string("SH_D"), this->humanNodes.back(), this->deskNodes.back()));
            }
        }
        else if (rel_type == 2)
        {
            int n_tries = 20;
            double angle = get_random_double(0, 2*M_PI);
            double angle_ = angle;
            // if (angle+M_PI>2*M_PI)
            // {
            //     angle_ = 2*M_PI - (angle+M_PI);
            // }
            // else
            // {
            //     angle_ = angle+M_PI;
            // }
            double ix, iz, ix_, iz_;
            std::vector<double> human_p = {0., 0.};
            
            double dist = 0.8 + (rand()%3)/10.0;
            double delta_x = dist*sin(angle), delta_y = dist*cos(angle);
            std::vector<double> plant_p = {human_p[0]+delta_x, human_p[1]+delta_y};
            do
            {
                this->place_the_object(human_p[0], human_p[1], 1.2);
                ix = (robot_p[0]-human_p[0]);
                iz = (robot_p[1]-human_p[1]);

                plant_p = {human_p[0]+delta_x, human_p[1]+delta_y};
                if ((plant_p[0] > -this->wall_length / 2.0 + 1.2) and (plant_p[0] < this->wall_length / 2.0 + 1.2) and (plant_p[1] > -this->wall_length / 2.0 + 1.2) and (plant_p[1] < this->wall_length / 2.0 + 1.2))
                {
                    ix_ = (robot_p[0]-plant_p[0]);
                    iz_ = (robot_p[1]-plant_p[1]);
                }
                else
                {
                    n_tries--;
                    continue;
                }
                n_tries--;
            } while ((sqrt(ix*ix + iz*iz) < 0.8 or sqrt(ix_*ix_ + iz_*iz_) < 0.8) and n_tries>0);

            if(n_tries>0)
            {
                printf("Static Human - Plant Relaton\n");
                this->humanNodes.push_back(this->wapi->create_humans(std::string("static_human_r_")+std::to_string(i), human_p[0], human_p[1], angle, "None"));
                this->plantNodes.push_back(this->wapi->create_object(std::string("plant_r_")+std::to_string(i), plant_p[0], plant_p[1], angle_, "plant"));
                this->relationNodes.push_back(this->wapi->create_relations(std::string("relation_")+std::to_string(i), std::string("SH_P"), this->humanNodes.back(), this->plantNodes.back()));
            }
        }
    }
}



void SpecificWorker::create_test_bed()
{
    printf("TEST BED!\n");
    double threshold = 0.5;
    std::vector<double> vect = {0., 0., 0.};
    int n_humans;

    if (this->level == 1)
        n_humans = 1;
    else if (this->level == 2)
        n_humans = 2;
    else if (this->level == 3)
        n_humans = 4;
    else if (this->level == 4)
        n_humans = 6;
    else if (this->level == 5)
        n_humans = 8;

    if (this->type == "square")
    {
        printf("SQUARE FORMATION OF MOVING HUMANS: %d\n", n_humans);
        for (int i=0; i<n_humans; i++)
        {
            int side = i%4;

            double ix, iz, angle, ix_g, iz_g;
            std::vector<double> human_p = {0., 0.};
            std::vector<double> human_goal = {0., 0.};
            int n_tries = 20;
            do
            {
                this->place_human_square(human_p[0], human_p[1], threshold, side, angle, human_goal[0], human_goal[1]);
                ix = (robot_p[0]-human_p[0]);
                iz = (robot_p[1]-human_p[1]);
                ix_g = (robot_p[0]-human_goal[0]);
                iz_g = (robot_p[1]-human_goal[1]);
                n_tries--;
            } while ((sqrt(ix*ix + iz*iz) < 0.8 or sqrt(ix_g*ix_g + iz_g*iz_g) < 0.8) and n_tries>0);

            if(n_tries>0)
            {
                this->human_goals.push_back(human_goal);
                this->humanNodes.push_back(this->wapi->create_humans(std::string("static_human_")+std::to_string(i), human_p[0], human_p[1], angle, "None"));
            }
        }
    }
    else if (this->type == "circle")
    {

        printf("CIRCLE FORMATION OF MOVING HUMANS: %d\n", n_humans);
        for (int i=0; i<n_humans; i++)
        {
            double ix, iz, angle, ix_g, iz_g;
            std::vector<double> human_p = {0., 0.};
            std::vector<double> human_goal = {0., 0.};
            int n_tries = 20;
            do
            {
                this->place_human_circle(human_p[0], human_p[1], threshold, angle, human_goal[0], human_goal[1]);
                ix = (robot_p[0]-human_p[0]);
                iz = (robot_p[1]-human_p[1]);
                ix_g = (robot_p[0]-human_goal[0]);
                iz_g = (robot_p[1]-human_goal[1]);
                n_tries--;
            } while ((sqrt(ix*ix + iz*iz) < 1.2 or sqrt(ix_g*ix_g + iz_g*iz_g) < 1.2) and n_tries>0);

            if(n_tries>0)
            {
                this->human_goals.push_back(human_goal);
                this->humanNodes.push_back(this->wapi->create_humans(std::string("static_human_")+std::to_string(i), human_p[0], human_p[1], angle, "None"));
            }
        }
    }
    else if (this->type == "stationary")
    {

        printf("STATIONARY HUMANS: %d\n", n_humans);
        for (int i=0; i<n_humans; i++)
        {
            double angle = get_random_double(-M_PIl, M_PIl);

            double ix, iz;
            std::vector<double> human_p = {0., 0.};
            int n_tries = 20;
            do
            {
                this->place_the_object(human_p[0], human_p[1], threshold);
                ix = (robot_p[0]-human_p[0]);
                iz = (robot_p[1]-human_p[1]);
                n_tries--;
            } while (sqrt(ix*ix + iz*iz) < 0.8 and n_tries>0);

            if(n_tries>0)
                this->humanNodes.push_back(this->wapi->create_humans(std::string("static_human_")+std::to_string(i), human_p[0], human_p[1], angle, "None"));
        }
    }
    else if (this->type == "random")
    {

        printf("RANDOM FORMATION OF MOVING HUMANS: %d\n", n_humans);
        for (int i=0; i<n_humans; i++)
        {
            double ix, iz, angle, ix_g, iz_g;
            std::vector<double> human_p = {0., 0.};
            std::vector<double> human_goal = {0., 0.};
            int n_tries = 20;
            do
            {
                this->place_human_random(human_p[0], human_p[1], angle, human_goal[0], human_goal[1]);
                ix = (robot_p[0]-human_p[0]);
                iz = (robot_p[1]-human_p[1]);
                ix_g = (robot_p[0]-human_goal[0]);
                iz_g = (robot_p[1]-human_goal[1]);
                n_tries--;
            } while ((sqrt(ix*ix + iz*iz) < 1.2 or sqrt(ix_g*ix_g + iz_g*iz_g) < 1.2) and n_tries>0);

            if(n_tries>0)
            {
                printf("HUMAN POSE %f %f\n", human_p[0], human_p[1]);
                printf("HUMAN GOAL POSE %f %f\n", human_goal[0], human_goal[1]);
                this->human_goals.push_back(human_goal);
                this->humanNodes.push_back(this->wapi->create_humans(std::string("static_human_")+std::to_string(i), human_p[0], human_p[1], angle, "None"));
            }
        }
    }
}