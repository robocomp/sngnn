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
#ifndef GENERICWORKER_H
#define GENERICWORKER_H

#include "config.h"
#include <stdint.h>


#if Qt5_FOUND
	#include <QtWidgets>
#else
	#include <QtGui>
#endif
#include <ui_mainUI.h>
#include <CommonBehavior.h>

#include <ByteSequencePublisher.h>
#include <GenericBase.h>
#include <GoalPublisher.h>
#include <InteractionDetector.h>
#include <ObjectDetector.h>
#include <OmniRobot.h>
#include <PeopleDetector.h>
#include <Simulator.h>
#include <WallDetector.h>


#define CHECK_PERIOD 5000
#define BASIC_PERIOD 100

#include <map>
using namespace std;

typedef map <string,::IceProxy::Ice::Object*> MapPrx;


class GenericWorker : public QWidget, public Ui_guiDlg
{
Q_OBJECT
public:
	GenericWorker(MapPrx& mprx);
	virtual ~GenericWorker();
	virtual void killYourSelf();
	virtual void setPeriod(int p);

	QMutex *mutex;


	RoboCompByteSequencePublisher::ByteSequencePublisherPrx bytesequencepublisher_pubproxy;
	RoboCompGoalPublisher::GoalPublisherPrx goalpublisher_pubproxy;
	RoboCompInteractionDetector::InteractionDetectorPrx interactiondetector_pubproxy;
	RoboCompObjectDetector::ObjectDetectorPrx objectdetector_pubproxy;
	RoboCompPeopleDetector::PeopleDetectorPrx peopledetector_pubproxy;
	RoboCompWallDetector::WallDetectorPrx walldetector_pubproxy;

	virtual void OmniRobot_correctOdometer(const int x, const int z, const float alpha) = 0;
	virtual void OmniRobot_getBasePose(int &x, int &z, float &alpha) = 0;
	virtual void OmniRobot_getBaseState(RoboCompGenericBase::TBaseState &state) = 0;
	virtual void OmniRobot_resetOdometer() = 0;
	virtual void OmniRobot_setOdometer(const RoboCompGenericBase::TBaseState &state) = 0;
	virtual void OmniRobot_setOdometerPose(const int x, const int z, const float alpha) = 0;
	virtual void OmniRobot_setSpeedBase(const float advx, const float advz, const float rot) = 0;
	virtual void OmniRobot_stopBase() = 0;
	virtual void Simulator_regenerate(const std::string &scene) = 0;
	virtual void Simulator_step(const int timestep) = 0;

protected:

	QTimer timer;
	int Period;

private:


public slots:
	virtual void compute() = 0;
	virtual void initialize(int period) = 0;
	
signals:
	void kill();
};

#endif
