# Poligonszerkesztő

- A poligon csúcsait sorban a bal egérgomb lenyomásával lehet definiálni.

- A jobb egérgom lenyomásakor a határon (a poligon élein) a kurzorhoz legközelebb felvesz egy új pontot (nem feltétlenül már létező csúcspont!), és az élszakasz kezdő és végpontja közé beköti. Amíg a jobb egérgombot el nem engedjük ez az új csúcs követi a kurzor mozgását, mialatt a poligon is deformálódik.

- A poligont az 's' billentyűvel lehet szebben görbülővé tenni, melynek algoritmusa a következő. A meglévő csúcspontokra egy egyenletes, -1-es tenziójú Catmull-Rom splinet illesztünk, és két meglévő pont közé a paramétereik számtani közepének megfelelő spline pontot szúrunk.


- A poligont az 'd' billentyűvel lehet egyszerűsíteni, melynek hatására a poligon csúcsainak száma megfeleződik. Azokat a csúcsokat dobjuk el, amelyek eldobása minimálisan módosítja a poligon alakját, azaz amelynél az eldobott csúcs és a keletkező új él távolsága minimális.


![alt text](https://github.com/levente-murgas/grafika1/blob/main/res/img.png?raw=true)
