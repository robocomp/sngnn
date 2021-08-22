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
#include "genericworker.h"
/**
* \brief Default constructor
*/
GenericWorker::GenericWorker(MapPrx& mprx) : Ui_guiDlg()
{

	bytesequencepublisher_pubproxy = (*(RoboCompByteSequencePublisher::ByteSequencePublisherPrx*)mprx["ByteSequencePublisherPub"]);
	goalpublisher_pubproxy = (*(RoboCompGoalPublisher::GoalPublisherPrx*)mprx["GoalPublisherPub"]);
	interactiondetector_pubproxy = (*(RoboCompInteractionDetector::InteractionDetectorPrx*)mprx["InteractionDetectorPub"]);
	objectdetector_pubproxy = (*(RoboCompObjectDetector::ObjectDetectorPrx*)mprx["ObjectDetectorPub"]);
	peopledetector_pubproxy = (*(RoboCompPeopleDetector::PeopleDetectorPrx*)mprx["PeopleDetectorPub"]);
	walldetector_pubproxy = (*(RoboCompWallDetector::WallDetectorPrx*)mprx["WallDetectorPub"]);

	mutex = new QMutex(QMutex::Recursive);


	#ifdef USE_QTGUI
		setupUi(this);
		show();
	#endif
	Period = BASIC_PERIOD;
	connect(&timer, SIGNAL(timeout()), this, SLOT(compute()));

}

/**
* \brief Default destructor
*/
GenericWorker::~GenericWorker()
{

}
void GenericWorker::killYourSelf()
{
	printf("Killing myself");
	emit kill();
}
/**
* \brief Change compute period
* @param per Period in ms
*/
void GenericWorker::setPeriod(int p)
{
	Period = p;
	timer.start(Period);
}
