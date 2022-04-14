//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
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
const char * const vertexSource = R"(
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
const char * const fragmentSource = R"(
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
    Camera2D() : wCenter(0, 0), wSize(200, 200) { }
    mat4 V() { return TranslateMatrix(-wCenter); }
    mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

    mat4 Vinv() { return TranslateMatrix(wCenter); }
    mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }

    void Zoom(float s) { wSize = wSize * s; }
    void Pan(vec2 t) { wCenter = wCenter + t; }
};

Camera2D camera;		// 2D camera
GPUProgram gpuProgram; // vertex and fragment shaders

class Circle{
public:
    unsigned int vao;	// vertex array object id
    float sx, sy;		// scaling
    vec2 wTranslate;	// translation
    float phi;			// angle of rotation
    double size;
    Circle() { Animate(0); }
    void create() {
        size = 3;
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
        gpuProgram.setUniform(vec3(1.0f, 0.0f, 0.0f), "color");
        // Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
    }
    void Animate(float t) {
        sx = 10;
        sy = 10;
        wTranslate = vec2(0, 0);
        phi = t;

    }

    mat4 M() {
        mat4 Mscale(sx, 0, 0, 0,
                    0, sy, 0, 0,
                    0, 0,  0, 0,
                    0, 0,  0, 1); // scaling

        mat4 Mrotate(cosf(phi), sinf(phi), 0, 0,
                     -sinf(phi), cosf(phi), 0, 0,
                     0,        0,        1, 0,
                     0,        0,        0, 1); // rotation

        mat4 Mtranslate(1,            0,            0, 0,
                        0,            1,            0, 0,
                        0,            0,            0, 0,
                        wTranslate.x, wTranslate.y, 0, 1); // translation

        return Mscale * Mrotate * Mtranslate;	// model transformation
    }

    void Draw() {
        // set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
        mat4 MVPTransform = M() * camera.V() * camera.P();
        gpuProgram.setUniform(MVPTransform, "MVP");
        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        glDrawArrays(GL_TRIANGLE_FAN, 0, 21);	// draw a single triangle with vertices defined in vao
    }
};
class LineStrip {
public:
    unsigned int		vao, vbo;	// vertex array object, vertex buffer object
    std::vector<vec2>   controlPoints; // interleaved data of coordinates and colors
    std::vector<float>  vertexData; // interleaved data of coordinates and colors
    vec2			    wTranslate; // translation

    void create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0
        glEnableVertexAttribArray(1);  // attribute array 1
        // Map attribute array 0 to the vertex data of the interleaved vbo
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
        // Map attribute array 1 to the color data of the interleaved vbo
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
    }

    mat4 M() { // modeling transform
        return mat4(1,            0,            0, 0,
                    0,            1,            0, 0,
                    0,            0,            1, 0,
                    wTranslate.x, wTranslate.y, 0, 1); // translation
    }

    mat4 Minv() { // inverse modeling transform
        return mat4(1,              0,            0, 0,
                    0,              1,            0, 0,
                    0,              0,            1, 0,
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


    void Draw() {
        if (vertexData.size() > 0) {
            // set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
            mat4 MVPTransform = M() * camera.V() * camera.P();
            gpuProgram.setUniform(MVPTransform, "MVP");
            glBindVertexArray(vao);
            glDrawArrays(GL_LINE_STRIP, 0, vertexData.size() / 5);
        }
    }
};

// The virtual world: collection of two objects
Circle circle;
LineStrip lineStrip;
// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight); 	// Position and size of the photograph on screen
    glLineWidth(2.0f); // Width of lines in pixels

    circle.create();
    lineStrip.create();
    lineStrip.AddPoint(0.0f,0.0f);
    lineStrip.AddPoint(2.0f,2.0f);

    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");

    printf("\nUsage: \n");
    printf("Mouse Left Button: Add control point to polyline\n");
    printf("Key 'a': Camera pan -x\n");
    printf("Key 'd': Camera pan +x\n");
    printf("Key 'w': Camera pan -y\n");
    printf("Key 's': Camera pan +y\n");
    printf("Key 'q': Camera zoom in\n");
    printf("Key 'e': Camera zoom out\n");
}


// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0.5, 0.5, 0.5, 0);							// background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    circle.Draw();
    lineStrip.Draw();
    glutSwapBuffers();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    switch (key) {
            case 'a': camera.Pan(vec2(-1, 0)); break;
            case 'd': camera.Pan(vec2(+1, 0)); break;
            case 'w': camera.Pan(vec2( 0, 1)); break;
            case 's': camera.Pan(vec2( 0,-1)); break;
            case 'q': camera.Zoom(0.9f); break;
            case 'e': camera.Zoom(1.1f); break;
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
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
       lineStrip.AddPoint(cosf(time) * 6,sinf(time) * 6);
        glutPostRedisplay();


}
