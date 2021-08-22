/*
 *    Copyright (C) 2021 by YOUR NAME HERE
 *
 *    This file is part of 
 *
 *     is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *     is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with .  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <list>

#include <genericworker.h>

#include <webots/Supervisor.hpp>

#include <WebotsAPI.h>
#include <RobotNode.h>

using namespace webots;

double get_random_double(const double &min_v, const double &max_v);
double get_random_int(const int &min_v, const int &max_v);

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
	SpecificWorker(MapPrx& mprx);
	~SpecificWorker();

	void OmniRobot_correctOdometer(const int x, const int z, const float alpha);
	void OmniRobot_getBasePose(int &x, int &z, float &alpha);
	void OmniRobot_getBaseState(RoboCompGenericBase::TBaseState &state);
	void OmniRobot_resetOdometer();
	void OmniRobot_setOdometer(const RoboCompGenericBase::TBaseState &state);
	void OmniRobot_setOdometerPose(const int x, const int z, const float alpha);
	void OmniRobot_setSpeedBase(const float advx, const float advz, const float rot);
	void OmniRobot_stopBase();
	void Simulator_regenerate(const std::string &scene);
	void Simulator_step(const int timestep=-1);

    void create_scene(const std::string &scene);
    void create_walls_and_floor(int walls);
	void create_robot_and_goal();
	void create_plants(const int &n_plants);
	void create_desks(const int &n_desks);
	void create_humans(const int &n_humans);
    void create_dyn_humans(const int &n_humans);
	void create_relations(const int &n_relations);
    void create_test_bed();

	void place_the_object(double &x, double &z, const double &threshold);
    void place_human_square(double &x, double &z, const double &threshold, int side, double &angle, double &x_g, double &z_g);
    void place_human_circle(double &x, double &z, const double &threshold, double &angle, double &x_g, double &z_g);
    void place_human_random(double &x, double &z, double &angle, double &x_g, double &z_g);

public slots:
	void compute();
	void initialize(int period);

private:

    bool RL_MODE;
    WebotsAPI *wapi;
    bool end_simulation;
    int HUMAN_INIT_MODE;
    std::string type;
    int level;

    RobotNode *robotNode = NULL;
	Goal *goalNode = NULL;

    float wall_length;
    std::list<Wall *> wallNodes;
    std::list<Node *> floorNodes;
    std::list<Obstacle *> plantNodes;
    std::list<Obstacle *> deskNodes;
    std::list<Human *> humanNodes;
    std::list<Relation *> relationNodes;
    std::vector<vector<double>> human_goals;

    

    std::unordered_map<WorldEntity *, std::vector<double>> coordinates, prev_coordinates;

protected:
    void stop_simulation() { this->end_simulation = true; }
    void clear_dynamic_checks();
    void publish_data();
    void clear_entities_list();
    void initialise_coordinates_map();
    void compute_current_coordinates();


    std::vector<double> robot_p = {0., 0.};

};

#endif
