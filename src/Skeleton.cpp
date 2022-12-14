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
// Nev    : Murg?s Levente
// Neptun : M5E2T9
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
	layout(location = 0) in vec2 vertexPosition;

	void main() {
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
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
    vec2 wCenter; // center in world coordinates
    vec2 wSize;   // width and height in world coordinates
public:
    Camera2D() : wCenter(0, 0), wSize(20, 20) { }

    mat4 V() { return TranslateMatrix(-wCenter); }
    mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

    mat4 Vinv() { return TranslateMatrix(wCenter); }
    mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }

    void Zoom(float s) { wSize = wSize * s; }
    void Pan(vec2 t) { wCenter = wCenter + t; }
};
Camera2D camera;		// 2D camera
GPUProgram gpuProgram; // vertex and fragment shaders

class MyPolygon {
    unsigned int vaoPoints, vboPoints;
protected:
    std::vector<vec2> wPoints;
    float tension = -1;
public:
    void create(){
        glGenVertexArrays(1,&vaoPoints);
        glBindVertexArray(vaoPoints);
        glGenBuffers(1,&vboPoints);
        glBindBuffer(GL_ARRAY_BUFFER,vboPoints);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE, sizeof(vec2), NULL);
    }

    void AddPoint(float cX, float cY) {
        vec4 wVertex = vec4(cX,cY,0,1) * camera.Pinv() * camera.Vinv();
        wPoints.push_back(vec2(wVertex.x,wVertex.y));
    }

    vec4 lerp(const vec4& p, const vec4& q, float t) {
        return p * (1 - t) + q * t;
    }

    int FindShortestDistPointIndex(vec2 wVertex){
        int N = wPoints.size();
        int p = 0;
        float shortest = 999;
        float dist;
        for (int i = 0; i < N; ++i){
            vec2 A = wPoints[i];
            vec2 B = wPoints[(N + i + 1) % N];
            vec2 AB = B - A;
            vec2 AP = wVertex - A;
            vec2 BP = wVertex - B;

            float projection = dot(AP,AB);
            float mod = dot(AB, AB);
            float d = projection / mod;

            if (d >= 1)
                dist = length(BP);
            else if (d <= 0)
                dist = length(AP);
            else {
                vec2 perpPoint = A + AB * d;
                vec2 ppP = wVertex - perpPoint;
                dist = length(ppP);
            }
            if (dist < shortest) {
                shortest = dist;
                p = i;
            }
        }
        return p;
    }

    int AddMovingPoint(float cX, float cY){
        if(wPoints.size() >= 2) {
            vec4 hVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
            vec2 wVertex = vec2(hVertex.x, hVertex.y);
            int p = FindShortestDistPointIndex(wVertex);
            wPoints.insert(wPoints.begin() + p + 1, wVertex);
            return p + 1;
        }
    }

    void HalvePoints(){
        int N = wPoints.size();
        std::vector<vec2> wPoints_new;
        std::vector<float> distances;
        float dist;
        for (int i = 0; i < N; i++) {
            vec2 A = wPoints[i];
            vec2 P = wPoints[(i + 1) % N];
            vec2 B = wPoints[(i + 2) % N];
            vec2 AB = B - A;
            vec2 AP = P - A;
            vec2 BP = P - B;

            float projection = dot(AP,AB);
            float mod = dot(AB, AB);
            float d = projection / mod;

            if (d >= 1)
                dist = length(BP);
            else if (d <= 0)
                dist = length(AP);
            else {
                vec2 perpPoint = A + AB * d;
                vec2 ppP = P - perpPoint;
                dist = length(ppP);
            }
            if(!distances.empty()) {
                bool smallest = true;
                for(int j = 0; j < distances.size(); j++) {
                    if(dist > distances[j]){
                        distances.insert(distances.begin() + j,dist);
                        wPoints_new.insert(wPoints_new.begin() + j,P);
                        smallest = false;
                        break;
                    }
                }
                if(smallest) {
                    distances.emplace_back(dist);
                    wPoints_new.emplace_back(P);
                }
            }
            else {
                distances.emplace_back(dist);
                wPoints_new.emplace_back(P);
            }
        }
        wPoints_new.erase(wPoints_new.begin(),wPoints_new.begin() + N/2);
        for(int i = 0; i < wPoints_new.size(); i++){
            for(int j = 0; j < wPoints.size(); j++){
                if(wPoints_new[i].x == wPoints[j].x && wPoints_new[i].y == wPoints[j].y){
                    wPoints.erase(wPoints.begin() + j);
                    break;
                }
            }
        }
    }

    vec2 Hermite(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t){
        float deltat = t1 - t0;
        t -= t0;
        float deltat2 = deltat * deltat;
        float deltat3 = deltat * deltat2;
        vec2 a0 = p0, a1 = v0;
        vec2 a2 = (p1 - p0) * 3 / deltat2 - (v1 + v0 * 2) / deltat;
        vec2 a3 = (p0 - p1) * 2 / deltat3 + (v1 + v0) / deltat2;
        return ((a3 * t + a2) * t + a1) * t + a0;
    }

    void CatmullRom() {
        std::vector<vec2> wPoints_new;
        int N = wPoints.size();
        for (int i = 0; i < N; ++i) {
            wPoints_new.emplace_back(wPoints[(N + i + 1) % N]);
            vec2 A = (wPoints[(N + i + 1) % N] - wPoints[(N + i + 0) % N]) / (i + 1 - i + 0);
            vec2 B = (wPoints[(N + i + 2) % N] - wPoints[(N + i + 1) % N]) / (i + 2 - i + 1);
            vec2 C = (wPoints[(N + i + 3) % N] - wPoints[(N + i + 2) % N]) / (i + 3 - i + 2);
            vec2 v0 = (A + B) * (float) ((1.0f - tension) / 2.0f);
            vec2 v1 = (B + C) * (float) ((1.0f - tension) / 2.0f);
            wPoints_new.emplace_back(Hermite(wPoints[(N + i + 1) % N], v0, i + 1, wPoints[(N + i + 2) % N], v1, i + 2, i + 1.5));
        }
        wPoints = wPoints_new;
    }

    void CR_spline() {
        int N = wPoints.size();
        std::vector<vec2> CR;
        for (int i = 0; i < N; ++i) {
            vec2 A = (wPoints[(N + i + 1) % N] - wPoints[(N + i + 0) % N]) / (i + 1 - i + 0);
            vec2 B = (wPoints[(N + i + 2) % N] - wPoints[(N + i + 1) % N]) / (i + 2 - i + 1);
            vec2 C = (wPoints[(N + i + 3) % N] - wPoints[(N + i + 2) % N]) / (i + 3 - i + 2);
            vec2 v0 = (A + B) * (float) ((1.0f - tension) / 2.0f);
            vec2 v1 = (B + C) * (float) ((1.0f - tension) / 2.0f);
            for (float t = i + 1; t < i + 2; t += 0.05)
                CR.emplace_back(Hermite(wPoints[(N + i + 1) % N], v0, i + 1, wPoints[(N + i + 2) % N], v1, i + 2, t));
        }
        static GLuint vbo = 0;
        if (vbo == 0)
            glGenBuffers(1,&vbo);
        glBindVertexArray(vaoPoints);
        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER,CR.size() * sizeof(vec2), &CR[0], GL_DYNAMIC_DRAW);
        gpuProgram.setUniform(vec3(0,0.5,1),"color");

        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE, sizeof(vec2), NULL);
        glDrawArrays(GL_LINE_STRIP,0,CR.size());
    }

    void MovePoint(int movingPoint, float cX, float cY){
        vec4 hVertex = vec4(cX,cY,0,1) * camera.Pinv() * camera.Vinv();
        wPoints[movingPoint] = vec2(hVertex.x,hVertex.y);
    }

    void Draw(){
       // CR_spline();
        mat4 VPTransform = camera.V() * camera.P();
        gpuProgram.setUniform(VPTransform, "MVP");

        glBindVertexArray(vaoPoints);
        glBindBuffer(GL_ARRAY_BUFFER,vboPoints);
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE, sizeof(vec2), NULL);
        glBufferData(GL_ARRAY_BUFFER,wPoints.size() * sizeof(vec2), &wPoints[0], GL_DYNAMIC_DRAW);

        if(wPoints.size() >= 2){
            gpuProgram.setUniform(vec3(1,1,1),"color");
            glDrawArrays(GL_LINE_LOOP,0,wPoints.size());
        }

        if(!wPoints.empty()) {
            gpuProgram.setUniform(vec3(1,0,0),"color");
            glPointSize(10.0f);
            glDrawArrays(GL_POINTS,0,wPoints.size());
        }

    }
};



MyPolygon polygon;



// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
    polygon.create();
	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer
    polygon.Draw();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 's') {
        polygon.CatmullRom();
        glutPostRedisplay();
    }
    else if (key == 'd') {
        polygon.HalvePoints();
        glutPostRedisplay();
    }
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}


int movingPoint = -1;

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        // Convert to normalized device space
        float cX = 2.0f * pX / windowWidth - 1;    // flip y axis
        float cY = 1.0f - 2.0f * pY / windowHeight;
        polygon.AddPoint(cX,cY);
        glutPostRedisplay();
    }

    if(button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN){
        // Convert to normalized device space
        float cX = 2.0f * pX / windowWidth - 1;    // flip y axis
        float cY = 1.0f - 2.0f * pY / windowHeight;
        movingPoint = polygon.AddMovingPoint(cX,cY);
        glutPostRedisplay();
    }

    if(button == GLUT_RIGHT_BUTTON && state == GLUT_UP){
        movingPoint = -1;
    }
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    // Convert to normalized device space
    float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
    float cY = 1.0f - 2.0f * pY / windowHeight;
    if(movingPoint >= 0) polygon.MovePoint(movingPoint,cX,cY);
    glutPostRedisplay();
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
