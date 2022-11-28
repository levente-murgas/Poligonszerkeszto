//=============================================================================================
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Murgás Levente
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
//---------------------------
    const char * vertexSource = R"(
		#version 330
		precision highp float;

        const float alpha = 0.25f;
        const float beta = 0.2f;
		uniform mat4  M;
        uniform float z, w;

		layout(location = 0) in vec4  vtxPos;            // pos in modeling space
		layout(location = 1) in vec4  drdU;
		layout(location = 2) in vec4  drdV;
		layout(location = 3) in vec2  vtxUV;

		out vec4 wNormal;		    // normal in world space
		out vec4 wView;             // view in world space
		out vec2 texcoord;

		void main() {
		    texcoord = vtxUV;
			vec4 p4d = vtxPos * M * 0.5f + vec4(0,0,1,2);
            vec3 n1 = cross(drdU.xyz, drdV.xyz) * w;
            vec3 n2 = cross(drdU.xyw, drdV.xyw) * z;
		    wNormal = vec4(n1.xy - n2.xy, n1.z, n2.z) * M;
		    wView  = -p4d;
			gl_Position = vec4(p4d.x, p4d.y, p4d.z * length(p4d) / 3, p4d.z);
		}
	)";

    // fragment shader in GLSL
    const char * fragmentSource = R"(
		#version 330
		precision highp float;

        const float shininess = 50;
        const vec3 Lin = vec3(10,10,10);
        const vec4 light = vec4(1, 1, -1, -1);
		uniform sampler2D diffuseTexture;

		in  vec4 wNormal;       // interpolated world sp normal
		in  vec4 wView;         // interpolated world sp view
		in  vec2 texcoord;

        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
            vec4 L = normalize(light);
			vec4 V = normalize(wView);
            vec4 N = normalize(wNormal);
            vec4 H = normalize(L + V);
            float cost = abs(dot(L,N)), cosd = abs(dot(H,N));
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;

            vec3 radiance = texColor * (cost + 0.1f) + Lin * pow(cosd, shininess);
			fragmentColor = vec4(radiance, 1);
		}
	)";

    GPUProgram gpuProgram;
    float z = 0.5, w  = 0.5;
    bool wireFrame = false;
    int tesselationLevel = 30;

//---------------------------
template<class T> struct Dnum { // Dual numbers for automatic derivation
//---------------------------
    float f; // function value
    T d;  // derivatives
    Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
    Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
    Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
    Dnum operator*(Dnum r) {
        return Dnum(f * r.f, f * r.d + d * r.f);
    }
    Dnum operator/(Dnum r) {
        return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
    }
};

// Elementary functions prepared for the chain rule as well
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f)*g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f)*g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f)*g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f)*g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f)*g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
    return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;


const int tessellationLevel = 30;

//---------------------------
struct Camera { // 3D camera
//---------------------------
    vec3 wEye, wLookat, wVup;   // extrinsic
    float fov, asp, fp, bp;		// intrinsic
public:
    Camera() {
        asp = (float)windowWidth / windowHeight;
        fov = 75.0f * (float)M_PI / 180.0f;
        fp = 1; bp = 20;
    }
    mat4 V() { // view matrix: translates the center to the origin
        vec3 w1 = normalize(wEye - wLookat);
        vec3 u = normalize(cross(wVup, w));
        vec3 v = cross(w, u);
        return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w1.x, 0,
                                                   u.y, v.y, w1.y, 0,
                                                   u.z, v.z, w1.z, 0,
                                                   0,   0,   0,   1);
    }

    mat4 P() { // projection matrix
        return mat4(1 / (tan(fov / 2)*asp), 0,                0,                      0,
                    0,                      1 / tan(fov / 2), 0,                      0,
                    0,                      0,                -(fp + bp) / (bp - fp), -1,
                    0,                      0,                -2 * fp*bp / (bp - fp),  0);
    }
};

//---------------------------
struct Material {
//---------------------------
    vec3 kd, ks, ka;
    float shininess;
};

//---------------------------
struct Light {
//---------------------------
    vec3 La, Le;
    vec4 wLightPos; // homogeneous coordinates, can be at ideal point
};

//---------------------------
class CheckerBoardTexture : public Texture {
//---------------------------
public:
    CheckerBoardTexture(const int width, const int height) : Texture() {
        glBindTexture(GL_TEXTURE_2D, textureId);
        std::vector<vec3> image(width * height);
        const vec3 green(1, 0, 0.7), purple(0, 1, 0.25);
        for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
                image[y * width + x] = (x & 1) ^ (y & 1) ? green : purple;
            }
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
};

//---------------------------
struct VertexData {
//---------------------------
    vec4 position, drdU, drdV;
    vec2 texcoord;
};

//---------------------------
class Geometry {
//---------------------------
protected:
    unsigned int vao;        // vertex array object
public:
    Geometry() {
        glGenVertexArrays(1, &vao);
    }
    virtual void Draw() = 0;
//    ~Geometry() {
//        glDeleteVertexArrays(1, &vao);
//    }
};

//---------------------------
class ParamSurface : public Geometry {
//---------------------------
    unsigned int nVtxPerStrip, nStrips;
public:
    ParamSurface() { nVtxPerStrip = nStrips = 0; }
    virtual VertexData GenVertexData(float u, float v) = 0;

    void create(int N = tessellationLevel, int M = tessellationLevel) {
        glBindVertexArray(vao);
        unsigned int vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        nVtxPerStrip = (M + 1) * 2;
        nStrips = N;
        std::vector<VertexData> vtxData;	// vertices on the CPU
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= M; j++) {
                vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
                vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
            }
        }
        glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
        glEnableVertexAttribArray(1);  // attribute array 1 = drdU
        glEnableVertexAttribArray(2);  // attribute array 2 = drdV
        glEnableVertexAttribArray(3); // attribute array 3 = texcoord
        // attribute array, components/attribute, component type, normalize?, stride, offset
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, drdU));
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, drdV));
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
    }

    void Draw() {
        glBindVertexArray(vao);
        for (unsigned int i = 0; i < nStrips; i++) {
            glDrawArrays((wireFrame) ? GL_LINE_STRIP : GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
        }
    }

    void Animate(float t1) {
        mat4 transform = mat4(
                            cos(t1 / 2), 0, sin(t1 / 2), 0,
                            0, 1, 0, 0,
                        -sin(t1 / 2), 0, cos(t1 / 2), 0,
                        0, 0, 0, 1) *
                         mat4(
                                 1, 0, 0, 0,
                                 0, cos(t1), 0, sin(t1),
                                 0, 0, 1, 0,
                                 0, -sin(t1), 0, cos(t1));

        gpuProgram.setUniform(transform,"M");
        gpuProgram.setUniform(z,"z");
        gpuProgram.setUniform(w,"w");
    }
};

vec4 cross(vec4 v1, vec4 v2) {
    vec3 v = cross(vec3(v1.x,v1.y,v1.z),vec3(v2.x,v2.y,v2.z));
    return vec4(v.x,v.y,v.z,0);
}

vec4 normalize(vec4 v) { return v * 1.0f / sqrtf(dot(v,v)); }

//---------------------------
class Klein : public ParamSurface {
//---------------------------
    const float R = 1, P = 0.5f, epsilon = 0.1f;
    Texture *texture;
    //const float size = 1.5f;
public:
    Klein() {
        create();
        texture = new CheckerBoardTexture(15,20);
    }
    VertexData GenVertexData(float u, float v) {
        VertexData vtxData;
        Dnum2 U(u * M_PI * 2, vec2(1, 0));
        Dnum2 V(v * M_PI * 2, vec2(0, 1));
        Dnum2 X = (Cos(U / 2) * Cos(V) - Sin(U / 2) * Sin(V * 2)) * R;
        Dnum2 Y = (Sin(U / 2) * Cos(V) + Cos(U / 2) * Sin(V * 2)) * R;
        Dnum2 Z = Cos(U) * (Sin(V) * epsilon + 1) * P;
        Dnum2 W = Sin(U) * (Sin(V) * epsilon + 1) * P;
        vtxData.position = vec4(X.f, Y.f, Z.f,W.f);
        vtxData.drdU = vec4 (X.d.x, Y.d.x, Z.d.x, W.d.x);
        vtxData.drdV = vec4 (X.d.y, Y.d.y, Z.d.y, W.d.y);
        vtxData.texcoord = vec2(u, v);
        return vtxData;
    }
};

Klein* klein;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    glLineWidth(2);
    klein = new Klein();
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    gpuProgram.create(vertexSource,fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 0);							// background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    klein->Draw();
    glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if(key == 32)
        wireFrame = !wireFrame;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
    z = (float) (pX - windowWidth / 2) / (windowWidth / 2);
    w = (float) (-pY + windowHeight / 2) / (windowHeight / 2);
}
// Mouse click event
void onMouse(int button, int state, int pX, int pY) { onMouseMotion(pX,pY); }

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    klein->Animate(glutGet(GLUT_ELAPSED_TIME) / 1000.0f);
    glutPostRedisplay();
}