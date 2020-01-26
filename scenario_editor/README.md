# SocNav - Social navigation dataset
This repository holds the files to generate the **SocNav1** social navigation dataset and a small script to test it using DGL.

The **SocNav1** dataset contains 5000 different social navigation scenarios where there is a robot in a room and people it might be interrupting. The goal would be to estimate to what extent the robot interrupts humans. Humans might be interacting with objects or other humans.

The variables that are used to evaluate how inconvenient the presence of the robot is are:
- The relative position of the robot wrt the people's perspective, the closer the worst.
- The number of people it is interrupting, the score would decrease as it increases.
- The closer it gets to the interaction lines (human to human, or human to object). The closer the lower the score.
- The density of humans in the room. In general terms, the higher the density, the more acceptable getting a bit closer is.

The format of the dataset is a single _socnav.json_ file, where each line holds the information for a single scenario using JSON. The format is the same for the labelled and unlabelled files, the only difference is that all unlabelled scenarios have a 0 score. The content of the different fields on the JSON structure is self-explanatory, and the criteria to label the scenarios is explained in section <a href="#understanding_and_contributing">Understanding & contributing to the dataset</a>.



## <a name="#using">Using the dataset</a>
- Software requirements.
- Where is the JSON file and how can it be used.
- Format of the file (agian).

## <a name="#understanding_and_contributing">Understanding & contributing to the dataset</a>



### English version

#### Requirements
To contribute to the dataset you will need the unlabelled dataset file (_socnav-unlabelled.json_) and a GNU/Linux system with Python 3.6 (or higher) and the following Python 3 modules:
- PySide 2
- NumPy

#### Running the program
To run the program we suggest to copy the input file to a temporary file that will be updated automatically by the script (You can use test.json here instead of socnav-unlabelled.json):
> cp socnav-unlabelled.json saved.json

The script is run as in the following command:
> python3 sndg.py INPUT_FILE.json >> OUTPUT_FILE.json

We suggest to use as input the temporary file and use any other file as the output file, for example, one named after you. The script will remove from the dataset the scenarios that you have assessed and write the remainder in a file called _saved.json_, that is why we suggest to use that very name for your dataset, so you do not need to modify the command used in later executions:
> python3 sndg.py saved.json >> my_name.json

You will be shown the GUI to label scenarios (see below to learn how to use it) and two files will be generated:
- OUTPUT_FILE.json: This file will contain the scenarios labelled. This is the file we need. **Mind the double '>' sign if you execute the script more than once!**
- saved.json: Labelling social navigation scenarios can be tiresome. To enable users do the task in multiple goes, the script will generate a file similar to the input file but *without* the scenarios that have already been labelled. This file can be used as input when resuming.

**To exit the application, close the window. Don't kill the script, as it wouldn't be able to generate the _saved.json_ file.**

#### Labelling social navigation scenarios
The following is a screenshot of the dataset labelling application:

![alt text](https://github.com/ljmanso/socnav/blob/master/img/ui.png)

Once running, to use the application to label scenarios, use the slider in the right hand side and click on "send context assessment". The program will generate the labelled samples by sending the text to the standard output (which might be redirected to a file as previously suggested).

Alternatively, instead of using the mouse to move the slider, you can use the keyboard arrows and page down/up keys to move the slider. You can also use the enter key to simulate clicking the social behaviour assessment button.

Please, consider the following guidelines when labelling the scenarios:

- The more people the robot could be interrupting or potentially making feel uncomfortable, the lower should be the score.

- According to works related to proxemics (the study of how humans use their personal space), the robot would be more likely to interrupt if it is in front of a person than at the back (people tend to walk forwards). Also, in general terms, the closer the robot is, the more disturbing.

- We want to consider, not only the personal spaces, but also the spaces that humans need to interact with other humans or objects, that is, we want to avoid the robot getting in the line of sight of people when interacting with object or other humans. The closer to the line of sight the lower the score.

- The density of humans in the room. In small rooms with a high number of people, closer distances are acceptable in comparison to big rooms with only one person. It is somewhat acceptable to get closer to people in crowded environments. Therefore, in general terms, the higher the density, the higher the score.

- If the robot is colliding with a human the label should be **unacceptable**.

- If the robot is colliding with an interaction's sight line the label should be **undesirable**.

- You should consider only **social** aspects, not robot's intelligence. Even if the robot seems to be having a (useless) close look at one of the walls, it should have a decent score as long as it is not disturbing anyone. We don't care whether or not the robot collides with walls and objects, as it is not related to social aspects.

- Even though we provide some guidelines, feel free to express how do you think you would feel about each particular situation if you were in the room with the robot.

### Spanish version

#### Requisitos
Para colaborar en la creación del _dataset_, necesitarás el fichero con los escenarios no etiquetados (_socnav-unlabelled.json_), un sistema GNU/Linux con Python 3.6 (o superior) y los siguientes módulos de Python 3:
- PySide 2
- NumPy

#### Ejecución del programa
Antes de ejecutar el programa, te sugerimos que copies el fichero de escenarios no etiquetados a un fichero temporal (_saved.json_) que será actualizado automáticamente por el programa:
> cp socnav-unlabelled.json saved.json

El programa se ejecuta mediante el siguiente comando:
> python3 sndg.py INPUT_FILE.json >> OUTPUT_FILE.json

Como fichero de entrada, debes usar el fichero temporal generado anteriormente (_saved.json_). El fichero de salida debe ser un fichero diferente, por ejemplo, un fichero que incluya tu nombre. El programa eliminará los escenarios que hayas etiquetado y almacenará los restantes en el fichero _saved.json_. Este es el motivo por el que sugerimos utilizar dicho fichero como fichero de entrada. De esta forma, no es necesario modificar el comando a ejecutar en posteriores ejecuciones:
> python3 sndg.py saved.json >> my_name.json

Una vez que se ejecuta el programa, se mostrará una interfaz gráfica que incluye las opciones necesarias para etiquetar los escenarios (más adelante se indica cómo usar esta interfaz). Tras la ejecución, se generarán dos ficheros:
- OUTPUT_FILE.json: Este fichero contiene los escenarios etiquetados. Es el fichero que necesitamos. **¡Recuerda incluir un doble '>' en la ejecución del comando si ejecutas el programa más de una vez!**
- saved.json: la tarea de etiquetado puede resultar tediosa. Por este motivo, el programa almacena los escenarios no etiquetados aún en este fichero. Utilizando este fichero como fichero de entrada, es posible ejecutar el programa tantas veces como se quiera para etiquetar nuevos escenarios, .

**Para finalizar la aplicación, cierra la ventana de la interfaz gráfica. No finalices el script de otra manera (comando _kill_, por ejemplo), ya que en ese caso el programa no podrá generar el fichero _saved.json_ .**

#### Etiquetado de escenarios para navegación social
La siguiente imagen muestra una captura de la aplicación de etiquetado:

![alt text](https://github.com/ljmanso/socnav/blob/master/img/ui.png)

Una vez en ejecución, para utilizar la aplicación para el etiquetado de escenarios, utiliza la barra de desplazamiento situada a la derecha para seleccionar la etiqueta más adecuada y, a continuación, haz clic en el botón "send context assessment". El programa generará las muestras etiquetadas enviando el texto a la salida estándar (que debe ser redirigida al fichero de salida tal y como se indicó anteriormente).

Alternativamente, en lugar de utilizar el ratón para mover la barra de desplazamiento, puedes usar las teclas de cursor, así como las teclas _RePag_ y _AvPag_. Asimismo, puede utilizarse la tecla _Enter_ como alternativa a la pulsación con el ratón del botón "send context assessment".

Para etiquetar los escenarios, puede tener en cuenta las indicaciones que se muestran a continuación:

- En los escenarios en los que hay más personas, es más probable que el robot interrumpa o incomode. En estos casos la puntuación debe ser menor que en escenarios donde no hay muchas personas.

- Teniendo en cuenta los trabajos relacionados con la proxémica (el estudio de cómo los humanos utilizan su espacio personal), es más probable que el robot interrumpa si se encuentra enfrente de una persona que si está a su espalda. Asimismo, en términos generales, cuanto más cerca se encuentra el robot de la persona más molestará.

- Queremos considerar no solo los espacios personales, sino también los espacios que los humanos necesitan para interactuar con otros humanos u objetos. Es decir, queremos evitar que el robot interrumpa la línea de visión de los humanos cuando estos interactúan con objetos o con otros humanos. Así, cuanto más cerca se encuentre el robot de la línea de visión de los humanos, menor debe ser la puntuación.

- La densidad de humanos en la habitación es otro aspecto a tener en cuenta. En habitaciones pequeñas con un número elevado de personas, distancias más cercanas son aceptables en comparación con habitaciones grandes con una sola persona. Así, puede considerarse adecuado acercarse más a las personas en espacios llenos. Por este motivo, en términos generales, cuanto mayor sea la densidad de personas, mayor debe ser también la puntuación.

- Independientemente de la situación, si el robot colisiona con un humano, la etiqueta debe ser **unacceptable**.

- Si el robot interrumpe el espacio de interacción de los humanos, la etiqueta debería ser **undesirable**.

- Debes considerar exclusivamente aspectos **sociales**, no la inteligencia del robot. Incluso si el robot se encuentra (inútilmente) cerca de una de las paredes, debe asignarse una puntuación alta si no molesta a ninguna de las personas presentes en la habitación. No importa si el robot colisiona o no con paredes u objetos, si la situación no está relacionada con ningún aspecto social.

- Estas indicaciones no pretenden forzar ninguna opinión. Siéntete libre para expresar cómo te sentirías en cada situación particular si tú fueras una de las personas presentes en la habitación con el robot.
