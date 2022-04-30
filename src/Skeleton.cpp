//=============================================================================================
// Mintaprogram: Z?ld h?romsz?g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    :
// Neptun :
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char *const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
        vec4 pos;
		pos = vec4(vp.x, vp.y, 0, 1) * MVP;	// transrm vp from modeling space to normalized device space
       pos.w = sqrt(pos.x * pos.x + pos.y * pos.y + 1)+1;
        gl_Position=pos;
	}
)";

// fragment shader in GLSL
const char *const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers

	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

// 2D camera
class Camera2D {
public:
    vec2 wCenter; // center in world coordinates
    vec2 wSize;   // width and height in world coordinates
    Camera2D() : wCenter(0, 0), wSize(200, 200) {}

    mat4 V() { return TranslateMatrix(-wCenter); }

    mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

    mat4 Vinv() { return TranslateMatrix(wCenter); }

    mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }

    void Zoom(float s) { wSize = wSize * s; }

    void Pan(vec2 t) { wCenter = wCenter + t; }
};

Camera2D camera;        // 2D camera
GPUProgram gpuProgram; // vertex and fragment shaders

class Circle {
public:
    unsigned int vao;    // vertex array object id
    float sx, sy;        // scaling
    vec2 wTranslate;    // translation
    vec3 color;
    float phi;            // angle of rotation
    double size;
    int weight;
    int power;
    mat4 MVPTransform;
    vec3 tcent;

    Circle() {}

    void create() {
        phi = 0;
        sx = 10;
        sy = 10;
        wTranslate = vec2(0, 0);
        tcent = vec3(0, 0, 0);
        MVPTransform = mat4(vec4(1, 0, 0, 0),
                            vec4(0, 1, 0, 0),
                            vec4(0, 0, 1, 0),
                            vec4(0, 0, 0, 1));
        power = rand() % 10000;
        while (power == 0) { power = rand() % 10000; }
        if (rand() % 2 == 0)
            power = power * -1;
        float colornum = ((float) power) / 10000;
        if (power < 0) {
            colornum = colornum * -1;
            color = vec3(colornum, 0.0f, 0.0f);
        } else
            color = vec3(0.0f, 0.0F, colornum);

        weight = rand() % 15000;
        while (weight == 0) { weight = rand() % 15000; }
        size = 1.5 + (((float) weight) / 5000);
        glGenVertexArrays(1, &vao);    // create 1 vertex array object
        glBindVertexArray(vao);        // make it active

        unsigned int vbo[2];        // vertex buffer objects
        glGenBuffers(2, &vbo[0]);    // Generate 2 vertex buffer objects

        // vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
        std::vector<float> dotpoints = {0.0f, 0.0f};
        for (int i = 0; i < 20; ++i) {
            float fi = i * 2 * M_PI / 19;
            dotpoints.push_back(cosf(fi) * size);
            dotpoints.push_back(sinf(fi) * size);
        }
        glBufferData(GL_ARRAY_BUFFER,    // Copy to GPU target
                     dotpoints.size() * sizeof(float),  // # bytes
                     &dotpoints[0],            // address
                     GL_STATIC_DRAW);    // we do not change later
        // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
        glEnableVertexAttribArray(0);
        // Data organization of Attribute Array 0
        glVertexAttribPointer(0,            // Attribute Array 0
                              2, GL_FLOAT,  // components/attribute, component type
                              GL_FALSE,        // not in fixed point format, do not normalized
                              0, NULL);     // stride and offset: it is tightly packed

        // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
        // Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
    }

    void Rotate(float t, vec3 tcent2) {
        tcent = tcent2;
        Draw(1);
        phi += t;
    }

    mat4 M(int imp) {
        if (imp == 0) {
            mat4 Mscale(sx, 0, 0, 0,
                        0, sy, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 1); // scaling

            mat4 Mrotate(cosf(phi), sinf(phi), 0, 0,
                         -sinf(phi), cosf(phi), 0, 0,
                         0, 0, 1, 0,
                         0, 0, 0, 1); // rotation

            mat4 Mtranslate(1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 0, 0,
                            wTranslate.x, wTranslate.y, 0, 1); // translation

            return Mscale * Mrotate * Mtranslate;
        }    // model transformation
        else if (imp == 1) {
            mat4 Mscale(sx, 0, 0, 0,
                        0, sy, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 1); // scaling
            mat4 Mtranslate2(1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 0, 0,
                             wTranslate.x - tcent.x, wTranslate.y - tcent.y, 0, 1); // translation

            mat4 Mrotate(cosf(phi), sinf(phi), 0, 0,
                         -sinf(phi), cosf(phi), 0, 0,
                         0, 0, 1, 0,
                         0, 0, 0, 1); // rotation

            mat4 Mtranslate(1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 0, 0,
                            tcent.x, tcent.y, 0, 1); // translation

            return Mscale * Mtranslate2 * Mrotate * Mtranslate;    // model transformation
        } else
            return M(0);
    }

    void Draw(int io) {
        // set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
        if (io == 0)
            MVPTransform = M(0) * camera.V() * camera.P();
        else if (io == 1) {
            MVPTransform = M(1) * camera.V() * camera.P();
        }
        gpuProgram.setUniform(MVPTransform, "MVP");
        gpuProgram.setUniform(color, "color");
        glBindVertexArray(vao);    // make the vao and its vbos active playing the role of the data source
        glDrawArrays(GL_TRIANGLE_FAN, 0, 21);    // draw a single triangle with vertices defined in vao
    }
};

class LineStrip {
public:
    float sx, sy, phi;
    unsigned int vao, vbo;    // vertex array object, vertex buffer object
    std::vector<vec2> controlPoints; // interleaved data of coordinates and colors
    std::vector<float> vertexData; // interleaved data of coordinates and colors
    vec2 wTranslate; // translation
    vec2 tcent;

    void create() {
        tcent = vec2(0, 0);
        wTranslate = vec2(0, 0);
        phi = 0;
        sx = 1;
        sy = 1;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        vertexData.clear();
        controlPoints.clear();
        glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0
        glEnableVertexAttribArray(1);  // attribute array 1
        // Map attribute array 0 to the vertex data of the interleaved vbo
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                              reinterpret_cast<void *>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
        // Map attribute array 1 to the color data of the interleaved vbo
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void *>(2 * sizeof(float)));
    }

    mat4 M(int imp) {
        if (imp == 0) {
            mat4 Mscale(sx, 0, 0, 0,
                        0, sy, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 1); // scaling

            mat4 Mrotate(cosf(phi), sinf(phi), 0, 0,
                         -sinf(phi), cosf(phi), 0, 0,
                         0, 0, 1, 0,
                         0, 0, 0, 1); // rotation

            mat4 Mtranslate(1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 0, 0,
                            wTranslate.x, wTranslate.y, 0, 1); // translation

            return Mscale * Mrotate * Mtranslate;
        }    // model transformation
        else if (imp == 1) {
            mat4 Mscale(sx, 0, 0, 0,
                        0, sy, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 1); // scaling
            mat4 Mtranslate2(1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 0, 0,
                             wTranslate.x - tcent.x, wTranslate.y - tcent.y, 0, 1); // translation

            mat4 Mrotate(cosf(phi), sinf(phi), 0, 0,
                         -sinf(phi), cosf(phi), 0, 0,
                         0, 0, 1, 0,
                         0, 0, 0, 1); // rotation

            mat4 Mtranslate(1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 0, 0,
                            tcent.x, tcent.y, 0, 1); // translation

            return Mscale * Mtranslate2 * Mrotate * Mtranslate;    // model transformation
        } else return M(0);
    }

    mat4 Minv() { // inverse modeling transform
        return mat4(1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    -wTranslate.x, -wTranslate.y, 0, 1); // inverse translation
    }

    void AddPoint(float cX, float cY) {
        // input pipeline
        vec4 mVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv() * Minv();

        controlPoints.push_back(vec2(mVertex.x, mVertex.y));
        // fill interleaved data
        vertexData.push_back(mVertex.x);
        vertexData.push_back(mVertex.y);
        vertexData.push_back(1); // red
        vertexData.push_back(1); // green
        vertexData.push_back(0); // blue
        // copy data to the GPU
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
    }

    void Draw(int io) {
        if (vertexData.size() > 0) {
            // set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
            int imp = 0;
            if (io == 1)
                imp = 1;
            mat4 MVPTransform = M(imp) * camera.V() * camera.P();

            gpuProgram.setUniform(MVPTransform, "MVP");
            gpuProgram.setUniform(vec3(1.0f, 1.0f, 1.0f), "color");
            glBindVertexArray(vao);
            glDrawArrays(GL_LINE_STRIP, 0, vertexData.size() / 5);
        }
    }

    void Rotate(float t, vec2 tcentr) {
        sx = 1;
        sy = 1;
        phi += t;
        tcent = tcentr;

    }
};

class Molecule {
public:
    float angularspeed;
    vec2 speed;
    float step;
    vec2 tcenter;
    float sx, sy, bufferx, buffery;    // scaling
    vec2 wTranslate;    // translation
    float phi;            // angle of rotation
    int weight;
    std::vector<vec2> controlPoints; // interleaved data of coordinates and colors
    std::vector<float> vertexData; // interleaved data of coordinates and colors
    std::vector<LineStrip> bonds;
    std::vector<Circle> atoms;

    void create(int shift) {
        angularspeed=0.0f;
        speed = vec2(0, 0);
        step = 0.0f;
        tcenter = vec2(0, 0);
        wTranslate = vec2(0, 0);
        sx = 0.0f;
        sy = 0.0f;
        bufferx = 0.0f;
        buffery = 0.0f;
        controlPoints.clear();
        weight = 0;
        vertexData.clear();
        bonds.clear();
        atoms.clear();
        phi = 0;
        bonds.clear();
        atoms.clear();
        int i = rand() % 8;
        while (i == 8) { i = rand() % 8; }
        // i =7;
        int atomcount = 9 - i;
        while (i < 9) {
            Circle circle;
            float x;
            if (rand() % 2 == 0)
                x = rand() % 200 * -1;
            else { x = rand() % 200; }
            float y;
            if (rand() % 2 == 0)
                y = rand() % 200 * -1;
            else y = rand() % 200;
            circle.create();
            circle.wTranslate.x = x + shift;
            circle.wTranslate.y = y + shift;
            atoms.push_back(circle);
            bufferx = x;
            buffery = y;
            i++;
        }

//        lineStrip.AddPoint(startcircle.wTranslate.x / 100, startcircle.wTranslate.y / 100);
//        lineStrip.AddPoint(atoms[0].wTranslate.x / 100, atoms[0].wTranslate.y / 100);
//        bonds.push_back(lineStrip);
        for (int f = 1; f < atomcount; ++f) {
            LineStrip lineStrip;
            lineStrip.create();
            Circle circlea = atoms[f];
            vec2 a(circlea.wTranslate.x / 100, circlea.wTranslate.y / 100);
            int g = rand() % f;
            Circle circleb = atoms[g];
            vec2 b(circleb.wTranslate.x / 100, circleb.wTranslate.y / 100);
            for (double j = 0; j < 1; j += 0.01) {

                vec2 c(j * b + (1 - j) * a);
                lineStrip.AddPoint(c.x, c.y);
            }
            bonds.push_back(lineStrip);
            vec2 xyw = vec2(0, 0);
            int allweight = 0;
            for (int j = 0; j < atomcount; ++j) {
                xyw = xyw + atoms[j].wTranslate * atoms[j].weight;
                allweight = allweight + atoms[j].weight;
            }
            tcenter = xyw / allweight;
            weight = allweight;
        }

    }

    void Move(vec2 cord) {
        for (int i = 0; i < atoms.size(); ++i) {
            atoms[i].wTranslate.x += cord.x;
            atoms[i].wTranslate.y += cord.y;
        }
        for (int i = 0; i < bonds.size(); ++i) {
            bonds[i].wTranslate.x += cord.x;
            bonds[i].wTranslate.y += cord.y;
        }
        tcenter.x += cord.x;
        tcenter.y += cord.y;
    }

    mat4 M() {
        mat4 Mscale(sx, 0, 0, 0,
                    0, sy, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 1); // scaling

        mat4 Mrotate(cosf(phi), sinf(phi), 0, 0,
                     -sinf(phi), cosf(phi), 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1); // rotation

        mat4 Mtranslate(1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 0, 0,
                        wTranslate.x, wTranslate.y, 0, 1); // translation

        return Mscale * Mrotate * Mtranslate;    // model transformation
    }

    void Draw() {
        mat4 MVPTransform = M() * camera.V() * camera.P();
        gpuProgram.setUniform(MVPTransform, "MVP");
        for (int i = 0; i < bonds.size(); ++i) {
            bonds[i].Draw(1);
        }
        for (int i = 0; i < atoms.size(); ++i) {
            atoms[i].Draw(1);
        }
    }

    void Rotate(float omega) {
        for (int i = 0; i < atoms.size(); ++i) {
            atoms[i].Rotate(omega, tcenter);
        }
        //bonds[i].wTranslate = vec2(tcenter);//ide megadom a tomegk?z?ppontot
        for (int i = 0; i < bonds.size(); ++i) {
            bonds[i].Rotate(omega, tcenter);
        }
    }
};

// The virtual world: collection of two objects
Molecule molecule;
Molecule molecule2;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);    // Position and size of the photograph on screen
    glLineWidth(2.0f); // Width of lines in pixels

    molecule.create(0);

    molecule2.create(200);
    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
    glClearColor(0.5, 0.5, 0.5, 0);                            // background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

    glutSwapBuffers();

    printf("\nUsage: \n");
    printf("Key 's': Camera pan -x\n");
    printf("Key 'd': Camera pan +x\n");
    printf("Key 'e': Camera pan -y\n");
    printf("Key 'x': Camera pan +y\n");
    printf("Key 'q': Camera zoom in\n");
    printf("Key 'w': Camera zoom out\n");
    printf("Key ' ': respawn atoms\n");

}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0.5, 0.5, 0.5, 0);                            // background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    molecule.Draw();
    molecule2.Draw();
    glutSwapBuffers();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 's')
        camera.Pan(vec2(-0.1, 0.0));
    else if (key == 'd')
        camera.Pan(vec2(0.1, 0));
    else if (key == 'e')
        camera.Pan(vec2(0, 0.1));
    else if (key == 'x')
        camera.Pan(vec2(0, -0.1));
    else if (key == 'q')
        camera.Zoom(0.9f);
    else if (key == 'w')
        camera.Zoom(1.1f);
    else if (key == ' ') {
        camera.wSize = vec2(200, 200);
        camera.wCenter = vec2(0, 0);
        molecule.create(0);
        molecule2.create(200);
    } else if (key == 'h') {
        molecule.Move(vec2(10, 10));
        molecule2.Move(vec2(-10, -10));
    } else if (key == 'g') {
        molecule.Move(vec2(-10, -10));
        molecule2.Move(vec2(10, 10));
    } else if (key == 'r') {
        molecule.Rotate(molecule.step);
        molecule.step = molecule.step + 0.1;
    }
    glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void fiz(Molecule &molecule, Molecule &molecule2, float dt) {
    for (int i = 0; i < molecule.atoms.size(); ++i) {
        for (int j = 0; j < molecule2.atoms.size(); ++j) {
            float culomb = (8.988e9 * molecule.atoms[i].power * 1.602e-19 * molecule2.atoms[j].power * 1.602e-19) /
                           (dot(molecule.atoms[i].wTranslate - molecule2.atoms[j].wTranslate,
                                molecule.atoms[i].wTranslate - molecule2.atoms[j].wTranslate));
            vec2 force = normalize(molecule2.atoms[j].wTranslate - molecule.atoms[i].wTranslate) * culomb;
            vec2 arm = molecule.atoms[i].wTranslate - molecule.tcenter;
            vec2 fullforce = dot(force, arm / length(arm)) * (arm / length(arm));
            vec2 acc = fullforce / molecule.weight / 1.6735e-27;
            molecule.speed = molecule.speed + acc * dt;
            vec2 rotForce = force - fullforce;
            float torque = rotForce.x * arm.y - rotForce.y * arm.x;
            float in = 0.0f;
            for (int j = 0; j < molecule.atoms.size(); j++) {
                in += 1.6735e-27 * dot(molecule.atoms[j].wTranslate - molecule.tcenter, molecule.atoms[j].wTranslate - molecule.tcenter);
            }
            molecule.angularspeed += torque / in * dt;

        }
        float torque2 = - molecule.angularspeed * 0.0000000000000000000000000000000000000000000001;
        float in = 0.0f;
        for (int j = 0; j < molecule.atoms.size(); j++) {
            in += 1.6735e-27 * dot(molecule.atoms[j].wTranslate - molecule.tcenter, molecule.atoms[j].wTranslate - molecule.tcenter);
        }
        molecule.angularspeed += torque2 / in * dt;




        vec2 force2 = -molecule.speed * 0.00000000000000000001;
        if(length(force2)>0.001){
            vec2 arm = molecule.atoms[i].wTranslate - molecule.tcenter;
            vec2 fullforce = dot(force2, arm / length(arm)) * (arm / length(arm));
            vec2 acc = fullforce / molecule.weight / 1.6735e-27;
            molecule.speed = molecule.speed + acc * dt;
            vec2 rotForce = force2 - fullforce;
            float torque = rotForce.x * arm.y - rotForce.y * arm.x;
            float in = 0.0f;
            for (int j = 0; j < molecule.atoms.size(); j++) {
                in += 1.6735e-27 * dot(molecule.atoms[j].wTranslate - molecule.tcenter, molecule.atoms[j].wTranslate - molecule.tcenter);
            }
            molecule.angularspeed += torque / in * dt;
        }}
    molecule.Move(molecule.speed * dt);
    molecule.Rotate(molecule.angularspeed/1.25 * dt);
}
void onIdle() {
    static float last = 0.0f;
    float time = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
    float elapsed = time - last;
    last = time;
    while (elapsed > 0) {
        float dt = std::min(elapsed, 0.01f);
        fiz(molecule, molecule2,dt);
        fiz(molecule2, molecule,dt);
        elapsed -= 0.01f;
    }

    glutPostRedisplay();
}
