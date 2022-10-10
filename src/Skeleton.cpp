//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
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
// Nev    : Murg�s Levente
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
const int nTesselatedVertices = 100;

class MyPolygon {
    unsigned int vaoPoints, vboPoints;
protected:
    std::vector<vec2> wPoints;
    std::vector<float> ts; //knots
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
        ts.push_back((float)wPoints.size() - 1);
        printf("Coordinates of %zu. point: x: %f y: %f\n", wPoints.size(),wVertex.x,wVertex.y);
    }

    vec4 lerp(const vec4& p, const vec4& q, float t) {
        return p * (1 - t) + q * t;
    }

    int AddMovingPoint(float cX, float cY){
        unsigned int pos = -1;
        if(wPoints.size() >= 2) {
            vec4 hVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
            vec2 wVertex = vec2(hVertex.x, hVertex.y);
            float shortest;
            float distance;
            vec4 closestOnSegment;
            for (unsigned int p = 0; p < wPoints.size(); p++) {
                vec4 p1 = vec4(wPoints[p].x, wPoints[p].y,0,1);
                //if p is the last we loop back to the first point with modulo size
                vec4 p2 = vec4(wPoints[(p + 1) % wPoints.size()].x, wPoints[(p + 1) % wPoints.size()].y,0,1);
                //distance of wVertex from segment
                for(int i = 0; i < 1000; i++){
                    float t = i / (1000 - 1.0f);
                    vec4 car = lerp(p1,p2,t);
                    distance = sqrt(dot(car - hVertex, car - hVertex));
                    if(p == 0 && i == 0){ //give shortest a starting value
                        shortest = distance;
                    }
                    if (distance < shortest) {
                        shortest = distance;
                        pos = p;
                        closestOnSegment = car;
                    }
                }
            }
            wPoints.insert(wPoints.begin() + pos + 1, vec2(wVertex.x, wVertex.y));
            ts.insert(ts.begin() + pos + 1,(float)(pos + 1));
            printf("distance = %f \n", distance);
            printf("The closest point on the segment was at x: %f y: %f\n",closestOnSegment.x,closestOnSegment.y);

        }
        return pos + 1;
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

    float tension = -1;

    vec2 r(float t){
        if(t <= ts[ts.size() - 1]) {
            for (int i = 0; i < wPoints.size() - 1; i++) {
                if (ts[i] <= t && t <= ts[i + 1]) {
                    vec2 vPrev = (i > 0) ? (wPoints[i] - wPoints[i - 1]) * (1.0f / (ts[i] - ts[i - 1])) : (wPoints[i] - wPoints[wPoints.size() - 1]) * (1.0f / (ts[i] - ts[wPoints.size() - 1]));
                    vec2 vCur = (wPoints[i + 1] - wPoints[i]) / (ts[i + 1] - ts[i]);
                    vec2 vNext = (i < wPoints.size() - 2) ? (wPoints[i + 2] - wPoints[i + 1]) / (ts[i + 2] - ts[i + 1]) : (wPoints[0] - wPoints[i + 1]) / (float)ts.size() - ts[i + 1];
                    vec2 v0 = (vPrev + vCur) * (float) ((1 - tension) / 2);
                    vec2 v1 = (vCur + vNext) * (float) ((1 - tension) / 2);
                    return Hermite(wPoints[i], v0, ts[i], wPoints[i + 1], v1, ts[i + 1], t);
                }
            }
        }
        else {
            int i = wPoints.size() - 1;
            vec2 vPrev = (wPoints[i] - wPoints[i - 1]) * (1.0f / (ts[i] - ts[i - 1]));
            vec2 vCur = (wPoints[0] - wPoints[i]) / ((float)ts.size()  - ts[i]);
            vec2 vNext = (wPoints[1] - wPoints[0]) / (ts[1] - (float)ts.size());
            vec2 v0 = (vPrev + vCur) * (float) ((1 - tension) / 2);
            vec2 v1 = (vCur + vNext) * (float) ((1 - tension) / 2);
            return Hermite(wPoints[i], v0, ts[i], wPoints[0], v1, (float)ts.size(), t);
        }
    }

    void CatmullRom(){
        if(wPoints.size() >= 4) {
            std::vector<vec2> vertexData;
            std::vector<float> knots;
            for (int i = 0; i < ts.size(); i++) {
                float t = (i < ts.size() - 1) ? (ts[i] + ts[i + 1]) / 2 : (ts[i] + (float)ts.size()) / 2;
                vec2 wVertex = r(t);
                vertexData.push_back(wVertex);
                knots.push_back(t);
            }
            for (int i = 0; i < vertexData.size(); i++) {
                for (int j = 0; j < wPoints.size() - 1; j++) {
                    if (ts[j] < knots[i] && knots[i] <= ts[j + 1]) {
                        wPoints.insert(wPoints.begin() + j + 1, vertexData[i]);
                        ts.insert(ts.begin() + j + 1, knots[i]);
                    }
                }
            }
            wPoints.push_back(vertexData[vertexData.size() - 1]);
            ts.push_back(knots[knots.size() - 1]);
        }
    }

    void MovePoint(int movingPoint, float cX, float cY){
        vec4 hVertex = vec4(cX,cY,0,1) * camera.Pinv() * camera.Vinv();
        wPoints[movingPoint] = vec2(hVertex.x,hVertex.y);
    }

    void Draw(){
        mat4 VPTransform = camera.V() * camera.P();
        gpuProgram.setUniform(VPTransform, "MVP");

        glBindVertexArray(vaoPoints);
        glBindBuffer(GL_ARRAY_BUFFER,vboPoints);
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
