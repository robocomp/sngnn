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


#include <signal.h>

// QT includes
#include <QtCore>
#include <QtGui>

// ICE includes
#include <Ice/Ice.h>
#include <IceStorm/IceStorm.h>
#include <Ice/Application.h>

#include <rapplication.h>
#include <sigwatch.h>

#include "config.h"
#include "genericmonitor.h"
#include "genericworker.h"
#include "specificworker.h"
#include "specificmonitor.h"
#include "commonbehaviorI.h"

#include <omnirobotI.h>
#include <simulatorI.h>

#include <GenericBase.h>



class simulator_supervisor_cpp : public RoboComp::Application
{
public:
	simulator_supervisor_cpp (QString prfx) { prefix = prfx.toStdString();}
private:
	void initialize();
	std::string prefix;
	MapPrx mprx;

public:
	virtual int run(int, char*[]);
};

void ::simulator_supervisor_cpp::initialize()
{
	// Config file properties read example
	// configGetString( PROPERTY_NAME_1, property1_holder, PROPERTY_1_DEFAULT_VALUE );
	// configGetInt( PROPERTY_NAME_2, property1_holder, PROPERTY_2_DEFAULT_VALUE );
}

int ::simulator_supervisor_cpp::run(int argc, char* argv[])
{
#ifdef USE_QTGUI
	QApplication a(argc, argv);  // GUI application
#else
	QCoreApplication a(argc, argv);  // NON-GUI application
#endif


	sigset_t sigs;
	sigemptyset(&sigs);
	sigaddset(&sigs, SIGHUP);
	sigaddset(&sigs, SIGINT);
	sigaddset(&sigs, SIGTERM);
	sigprocmask(SIG_UNBLOCK, &sigs, 0);

	UnixSignalWatcher sigwatch;
	sigwatch.watchForSignal(SIGINT);
	sigwatch.watchForSignal(SIGTERM);
	QObject::connect(&sigwatch, SIGNAL(unixSignal(int)), &a, SLOT(quit()));

	int status=EXIT_SUCCESS;

	RoboCompByteSequencePublisher::ByteSequencePublisherPrx bytesequencepublisher_pubproxy;
	RoboCompGoalPublisher::GoalPublisherPrx goalpublisher_pubproxy;
	RoboCompInteractionDetector::InteractionDetectorPrx interactiondetector_pubproxy;
	RoboCompObjectDetector::ObjectDetectorPrx objectdetector_pubproxy;
	RoboCompPeopleDetector::PeopleDetectorPrx peopledetector_pubproxy;
	RoboCompWallDetector::WallDetectorPrx walldetector_pubproxy;

	string proxy, tmp;
	initialize();

	sleep(2);

	IceStorm::TopicManagerPrx topicManager;
	try
	{
		topicManager = IceStorm::TopicManagerPrx::checkedCast(communicator()->propertyToProxy("TopicManager.Proxy"));
		if (!topicManager)
		{
		    cout << "[" << PROGRAM_NAME << "]: TopicManager.Proxy not defined in config file."<<endl;
		    cout << "	 Config line example: TopicManager.Proxy=IceStorm/TopicManager:default -p 9999"<<endl;
	        return EXIT_FAILURE;
		}
	}
	catch (const Ice::Exception &ex)
	{
		cout << "[" << PROGRAM_NAME << "]: Exception: 'rcnode' not running: " << ex << endl;
		return EXIT_FAILURE;
	}
	IceStorm::TopicPrx bytesequencepublisher_topic;

	while (!bytesequencepublisher_topic)
	{
		try
		{
			bytesequencepublisher_topic = topicManager->retrieve("ByteSequencePublisher");
		}
		catch (const IceStorm::NoSuchTopic&)
		{
			cout << "[" << PROGRAM_NAME << "]: Creating ByteSequencePublisher topic. \n";
			try
			{
				bytesequencepublisher_topic = topicManager->create("ByteSequencePublisher");
			}
			catch (const IceStorm::TopicExists&){
				// Another client created the topic.
				cout << "[" << PROGRAM_NAME << "]: ERROR publishing the ByteSequencePublisher topic. It's possible that other component have created\n";
			}
		}
		catch(const IceUtil::NullHandleException&)
		{
			cout << "[" << PROGRAM_NAME << "]: ERROR TopicManager is Null. Check that your configuration file contains an entry like:\n"<<
			"\t\tTopicManager.Proxy=IceStorm/TopicManager:default -p <port>\n";
			return EXIT_FAILURE;
		}
	}

	Ice::ObjectPrx bytesequencepublisher_pub = bytesequencepublisher_topic->getPublisher()->ice_oneway();
	bytesequencepublisher_pubproxy = RoboCompByteSequencePublisher::ByteSequencePublisherPrx::uncheckedCast(bytesequencepublisher_pub);
	mprx["ByteSequencePublisherPub"] = (::IceProxy::Ice::Object*)(&bytesequencepublisher_pubproxy);
	IceStorm::TopicPrx goalpublisher_topic;

	while (!goalpublisher_topic)
	{
		try
		{
			goalpublisher_topic = topicManager->retrieve("GoalPublisher");
		}
		catch (const IceStorm::NoSuchTopic&)
		{
			cout << "[" << PROGRAM_NAME << "]: Creating GoalPublisher topic. \n";
			try
			{
				goalpublisher_topic = topicManager->create("GoalPublisher");
			}
			catch (const IceStorm::TopicExists&){
				// Another client created the topic.
				cout << "[" << PROGRAM_NAME << "]: ERROR publishing the GoalPublisher topic. It's possible that other component have created\n";
			}
		}
		catch(const IceUtil::NullHandleException&)
		{
			cout << "[" << PROGRAM_NAME << "]: ERROR TopicManager is Null. Check that your configuration file contains an entry like:\n"<<
			"\t\tTopicManager.Proxy=IceStorm/TopicManager:default -p <port>\n";
			return EXIT_FAILURE;
		}
	}

	Ice::ObjectPrx goalpublisher_pub = goalpublisher_topic->getPublisher()->ice_oneway();
	goalpublisher_pubproxy = RoboCompGoalPublisher::GoalPublisherPrx::uncheckedCast(goalpublisher_pub);
	mprx["GoalPublisherPub"] = (::IceProxy::Ice::Object*)(&goalpublisher_pubproxy);
	IceStorm::TopicPrx interactiondetector_topic;

	while (!interactiondetector_topic)
	{
		try
		{
			interactiondetector_topic = topicManager->retrieve("InteractionDetector");
		}
		catch (const IceStorm::NoSuchTopic&)
		{
			cout << "[" << PROGRAM_NAME << "]: Creating InteractionDetector topic. \n";
			try
			{
				interactiondetector_topic = topicManager->create("InteractionDetector");
			}
			catch (const IceStorm::TopicExists&){
				// Another client created the topic.
				cout << "[" << PROGRAM_NAME << "]: ERROR publishing the InteractionDetector topic. It's possible that other component have created\n";
			}
		}
		catch(const IceUtil::NullHandleException&)
		{
			cout << "[" << PROGRAM_NAME << "]: ERROR TopicManager is Null. Check that your configuration file contains an entry like:\n"<<
			"\t\tTopicManager.Proxy=IceStorm/TopicManager:default -p <port>\n";
			return EXIT_FAILURE;
		}
	}

	Ice::ObjectPrx interactiondetector_pub = interactiondetector_topic->getPublisher()->ice_oneway();
	interactiondetector_pubproxy = RoboCompInteractionDetector::InteractionDetectorPrx::uncheckedCast(interactiondetector_pub);
	mprx["InteractionDetectorPub"] = (::IceProxy::Ice::Object*)(&interactiondetector_pubproxy);
	IceStorm::TopicPrx objectdetector_topic;

	while (!objectdetector_topic)
	{
		try
		{
			objectdetector_topic = topicManager->retrieve("ObjectDetector");
		}
		catch (const IceStorm::NoSuchTopic&)
		{
			cout << "[" << PROGRAM_NAME << "]: Creating ObjectDetector topic. \n";
			try
			{
				objectdetector_topic = topicManager->create("ObjectDetector");
			}
			catch (const IceStorm::TopicExists&){
				// Another client created the topic.
				cout << "[" << PROGRAM_NAME << "]: ERROR publishing the ObjectDetector topic. It's possible that other component have created\n";
			}
		}
		catch(const IceUtil::NullHandleException&)
		{
			cout << "[" << PROGRAM_NAME << "]: ERROR TopicManager is Null. Check that your configuration file contains an entry like:\n"<<
			"\t\tTopicManager.Proxy=IceStorm/TopicManager:default -p <port>\n";
			return EXIT_FAILURE;
		}
	}

	Ice::ObjectPrx objectdetector_pub = objectdetector_topic->getPublisher()->ice_oneway();
	objectdetector_pubproxy = RoboCompObjectDetector::ObjectDetectorPrx::uncheckedCast(objectdetector_pub);
	mprx["ObjectDetectorPub"] = (::IceProxy::Ice::Object*)(&objectdetector_pubproxy);
	IceStorm::TopicPrx peopledetector_topic;

	while (!peopledetector_topic)
	{
		try
		{
			peopledetector_topic = topicManager->retrieve("PeopleDetector");
		}
		catch (const IceStorm::NoSuchTopic&)
		{
			cout << "[" << PROGRAM_NAME << "]: Creating PeopleDetector topic. \n";
			try
			{
				peopledetector_topic = topicManager->create("PeopleDetector");
			}
			catch (const IceStorm::TopicExists&){
				// Another client created the topic.
				cout << "[" << PROGRAM_NAME << "]: ERROR publishing the PeopleDetector topic. It's possible that other component have created\n";
			}
		}
		catch(const IceUtil::NullHandleException&)
		{
			cout << "[" << PROGRAM_NAME << "]: ERROR TopicManager is Null. Check that your configuration file contains an entry like:\n"<<
			"\t\tTopicManager.Proxy=IceStorm/TopicManager:default -p <port>\n";
			return EXIT_FAILURE;
		}
	}

	Ice::ObjectPrx peopledetector_pub = peopledetector_topic->getPublisher()->ice_oneway();
	peopledetector_pubproxy = RoboCompPeopleDetector::PeopleDetectorPrx::uncheckedCast(peopledetector_pub);
	mprx["PeopleDetectorPub"] = (::IceProxy::Ice::Object*)(&peopledetector_pubproxy);
	IceStorm::TopicPrx walldetector_topic;

	while (!walldetector_topic)
	{
		try
		{
			walldetector_topic = topicManager->retrieve("WallDetector");
		}
		catch (const IceStorm::NoSuchTopic&)
		{
			cout << "[" << PROGRAM_NAME << "]: Creating WallDetector topic. \n";
			try
			{
				walldetector_topic = topicManager->create("WallDetector");
			}
			catch (const IceStorm::TopicExists&){
				// Another client created the topic.
				cout << "[" << PROGRAM_NAME << "]: ERROR publishing the WallDetector topic. It's possible that other component have created\n";
			}
		}
		catch(const IceUtil::NullHandleException&)
		{
			cout << "[" << PROGRAM_NAME << "]: ERROR TopicManager is Null. Check that your configuration file contains an entry like:\n"<<
			"\t\tTopicManager.Proxy=IceStorm/TopicManager:default -p <port>\n";
			return EXIT_FAILURE;
		}
	}

	Ice::ObjectPrx walldetector_pub = walldetector_topic->getPublisher()->ice_oneway();
	walldetector_pubproxy = RoboCompWallDetector::WallDetectorPrx::uncheckedCast(walldetector_pub);
	mprx["WallDetectorPub"] = (::IceProxy::Ice::Object*)(&walldetector_pubproxy);

	SpecificWorker *worker = new SpecificWorker(mprx);
	//Monitor thread
	SpecificMonitor *monitor = new SpecificMonitor(worker,communicator());
	QObject::connect(monitor, SIGNAL(kill()), &a, SLOT(quit()));
	QObject::connect(worker, SIGNAL(kill()), &a, SLOT(quit()));
	monitor->start();

	if ( !monitor->isRunning() )
		return status;

	while (!monitor->ready)
	{
		usleep(10000);
	}

	try
	{
		try {
			// Server adapter creation and publication
			if (not GenericMonitor::configGetString(communicator(), prefix, "CommonBehavior.Endpoints", tmp, "")) {
				cout << "[" << PROGRAM_NAME << "]: Can't read configuration for proxy CommonBehavior\n";
			}
			Ice::ObjectAdapterPtr adapterCommonBehavior = communicator()->createObjectAdapterWithEndpoints("commonbehavior", tmp);
			CommonBehaviorI *commonbehaviorI = new CommonBehaviorI(monitor);
			adapterCommonBehavior->add(commonbehaviorI, Ice::stringToIdentity("commonbehavior"));
			adapterCommonBehavior->activate();
		}
		catch(const Ice::Exception& ex)
		{
			status = EXIT_FAILURE;

			cout << "[" << PROGRAM_NAME << "]: Exception raised while creating CommonBehavior adapter: " << endl;
			cout << ex;

		}



		try
		{
			// Server adapter creation and publication
			if (not GenericMonitor::configGetString(communicator(), prefix, "OmniRobot.Endpoints", tmp, ""))
			{
				cout << "[" << PROGRAM_NAME << "]: Can't read configuration for proxy OmniRobot";
			}
			Ice::ObjectAdapterPtr adapterOmniRobot = communicator()->createObjectAdapterWithEndpoints("OmniRobot", tmp);
			OmniRobotI *omnirobot = new OmniRobotI(worker);
			adapterOmniRobot->add(omnirobot, Ice::stringToIdentity("omnirobot"));
			adapterOmniRobot->activate();
			cout << "[" << PROGRAM_NAME << "]: OmniRobot adapter created in port " << tmp << endl;
		}
		catch (const IceStorm::TopicExists&){
			cout << "[" << PROGRAM_NAME << "]: ERROR creating or activating adapter for OmniRobot\n";
		}


		try
		{
			// Server adapter creation and publication
			if (not GenericMonitor::configGetString(communicator(), prefix, "Simulator.Endpoints", tmp, ""))
			{
				cout << "[" << PROGRAM_NAME << "]: Can't read configuration for proxy Simulator";
			}
			Ice::ObjectAdapterPtr adapterSimulator = communicator()->createObjectAdapterWithEndpoints("Simulator", tmp);
			SimulatorI *simulator = new SimulatorI(worker);
			adapterSimulator->add(simulator, Ice::stringToIdentity("simulator"));
			adapterSimulator->activate();
			cout << "[" << PROGRAM_NAME << "]: Simulator adapter created in port " << tmp << endl;
		}
		catch (const IceStorm::TopicExists&){
			cout << "[" << PROGRAM_NAME << "]: ERROR creating or activating adapter for Simulator\n";
		}


		// Server adapter creation and publication
		cout << SERVER_FULL_NAME " started" << endl;

		// User defined QtGui elements ( main window, dialogs, etc )

		#ifdef USE_QTGUI
			//ignoreInterrupt(); // Uncomment if you want the component to ignore console SIGINT signal (ctrl+c).
			a.setQuitOnLastWindowClosed( true );
		#endif
		// Run QT Application Event Loop
		a.exec();


		status = EXIT_SUCCESS;
	}
	catch(const Ice::Exception& ex)
	{
		status = EXIT_FAILURE;

		cout << "[" << PROGRAM_NAME << "]: Exception raised on main thread: " << endl;
		cout << ex;

	}
	#ifdef USE_QTGUI
		a.quit();
	#endif

	status = EXIT_SUCCESS;
	monitor->terminate();
	monitor->wait();
	delete worker;
	delete monitor;
	return status;
}

int main(int argc, char* argv[])
{
	string arg;

	// Set config file
	QString configFile("etc/config");
	QString prefix("");
	if (argc > 1)
	{
	    QString initIC = QString("--Ice.Config=");
	    for (int i = 1; i < argc; ++i)
		{
		    arg = argv[i];
            if (arg.find(initIC.toStdString(), 0) == 0)
            {
                configFile = QString::fromStdString(arg).remove(0, initIC.size());
            }
            else
            {
                configFile = QString::fromStdString(argv[1]);
            }
        }

        // Search in argument list for --prefix= argument (if exist)
        QString prfx = QString("--prefix=");
        for (int i = 2; i < argc; ++i)
        {
            arg = argv[i];
            if (arg.find(prfx.toStdString(), 0) == 0)
            {
                prefix = QString::fromStdString(arg).remove(0, prfx.size());
                if (prefix.size()>0)
                    prefix += QString(".");
                printf("Configuration prefix: <%s>\n", prefix.toStdString().c_str());
            }
        }

	}
	::simulator_supervisor_cpp app(prefix);

	return app.main(argc, argv, configFile.toLocal8Bit().data());
}
