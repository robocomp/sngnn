cmake_minimum_required(VERSION 2.8.11)

INCLUDE( qt.cmake )

# Sources set
SET ( SOURCES
  specificworker.cpp
  specificmonitor.cpp
  rapplication.cpp
  sigwatch.cpp
  main.cpp
  genericmonitor.cpp
  commonbehaviorI.cpp
  genericworker.cpp
  omnirobotI.cpp
  simulatorI.cpp
  WebotsAPI.cpp
  Wall.cpp
  Goal.cpp
  RobotNode.cpp
  Obstacle.cpp
  WorldEntity.cpp
  Human.cpp
  Relation.cpp
)

SET (ICE_FILES ByteSequencePublisher GenericBase InteractionDetector ObjectDetector PeopleDetector WallDetector CommonBehavior GoalPublisher JoystickAdapter OmniRobot Simulator)

FOREACH( input_file ${ICE_FILES} )
    SET (SLICE_FILE_FOUND "false")
    MESSAGE(STATUS "Adding rule to generate ${input_file}.cpp and ${input_file}.h from ${input_file}.ice  (slice2cpp)" )
    ADD_CUSTOM_COMMAND (
        OUTPUT ${input_file}.cpp ${input_file}.h
        COMMAND slice2cpp --underscore -I. ${input_file}.ice --output-dir .
        DEPENDS ${input_file}.ice
        COMMENT "Generating ${input_file}.cpp and ${input_file}.h from ${input_file}.ice"
    )
    SET ( SOURCES ${SOURCES} ${input_file}.cpp)
    SET_PROPERTY(SOURCE ${input_file}.cpp PROPERTY SKIP_AUTOGEN ON)
ENDFOREACH( input_file )



# Headers set
SET ( HEADERS
  specificworker.h
  specificmonitor.h
)

INCLUDE_DIRECTORIES(/usr/local/webots/include/controller/cpp . ..)

SET(CMAKE_CXX_STANDARD 17)


link_directories( /usr/local/webots/lib/controller)

SET (LIBS ${LIBS} -lController -lCppController)



SET (EXECUTABLE_OUTPUT_PATH ..)

QT_WRAP_UI( UI_HEADERS mainUI.ui )


FIND_PACKAGE( Ice REQUIRED COMPONENTS Ice IceStorm OPTIONAL_COMPONENTS IceUtil )
FIND_PACKAGE( Threads)

SET( LIBS ${LIBS}  ${OPENTHREADS_LIBRARY} ${CMAKE_THREAD_LIBS_INIT}  -lboost_system  )



# Specify construction and link process
ADD_EXECUTABLE( simulator_supervisor_cpp ${SOURCES} ${MOC_SOURCES} ${RC_SOURCES} ${UI_HEADERS} )
TARGET_LINK_LIBRARIES( simulator_supervisor_cpp ${LIBS} ${STATIC_LIBS} ${SPECIFIC_LIBS} ${QT_LIBRARIES} ${Ice_LIBRARIES})

