//=============================================================================================
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

bool isInside(vec3 normal, vec3 p, vec3 r1, vec3 r2){
    return dot(cross(r2-r1,p-r1),normal) > 0;
}

struct Material {
    vec3 ka, kd, ks, n, kappa;
    float  shininess;
    vec3 F0;
    int rough, reflective, refractive;
};

struct RoughMaterial : Material {
    RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
        ka = _kd * M_PI;
        kd = _kd;
        ks = _ks;
        shininess = _shininess;
        rough = true;
        reflective = false;
        refractive = false;
    }
};

inline vec3 operator/(const vec3& v1, const vec3& v2) { return vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z); }

struct RefractiveAndRoughMaterial : Material {
    RefractiveAndRoughMaterial(vec3 _n, vec3 _kappa,vec3 _kd, vec3 _ks, float _shininess) {
        n = _n;
        kappa = _kappa;
        ka = _kd * M_PI;
        kd = _kd;
        ks = _ks;
        shininess = _shininess;
        vec3 one(1, 1, 1);
        F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
        rough = true;
        reflective = false;
        refractive = true;
    }
};

struct Hit {
    float t;
    vec3 position, normal;
    Material* material;
    Hit() { t = -1; }
};

struct Ray {
    vec3 start, dir;
    bool out = true;
    Ray(vec3 _start = vec3(), vec3 _dir = vec3(), bool _out = true) {
        start = _start;
        dir = normalize(_dir);
        out = _out;
    }
};

class Intersectable {
protected:
    Material* material;
public:
    virtual Hit intersect(const Ray& ray) = 0;
};

struct Cube : public Intersectable {
    struct face4 {
        unsigned int i, j, k, l;
        face4(unsigned int _i = 0, unsigned int _j = 0, unsigned int _k = 0, unsigned int _l = 0)
                : i(_i - 1), j(_j - 1), k(_k - 1), l(_l - 1){}
    };
    std::vector<vec3> vertices; //8 vertices
    std::vector<face4> faces; //6 faces

    Cube(const std::vector<vec3>& _vertices, const std::vector<face4>& _faces, Material* _material) {
        vertices = _vertices;
        faces = _faces;
        material = _material;
    }

    Hit intersect(const Ray& ray){
        Hit hit;
        for (const face4& face: faces) {
            vec3 p1 = vertices[face.i];
            vec3 p2 = vertices[face.j];
            vec3 p3 = vertices[face.k];
            vec3 u = p2 - p1;
            vec3 v = p3 - p1;
            vec3 normal = normalize(cross(u, v));
            float t = dot(normal,(p1 - ray.start)) / dot(normal, ray.dir);

            if ((t > 0 && hit.t == -1) || (t > 0 && t < hit.t)) { //find the plane that the ray intersects first
                vec3 p = ray.start + ray.dir * t;
                vec3 A = vertices[face.i];
                vec3 B = vertices[face.j];
                vec3 C = vertices[face.k];
                vec3 D = vertices[face.l];

                bool b1 = isInside(normal,p,A,B);
                bool b2 = isInside(normal,p,B,C);
                bool b3 = isInside(normal,p,C,D);
                bool b4 = isInside(normal,p,D,A);

                if (b1 && b2 && b3 && b4) {
                    hit.t = t;
                    hit.position = p;
                    hit.normal = normal;
                    hit.material = material;
                }
            }
        }
        return hit;
    }
};

struct Dodecahedron : public Intersectable {
    struct face5 {
        unsigned int i, j, k, l, m;
        face5(unsigned int _i = 0, unsigned int _j = 0, unsigned int _k = 0, unsigned int _l = 0, unsigned int _m = 0)
                : i(_i - 1), j(_j - 1), k(_k - 1), l(_l - 1), m(_m - 1) {}
    };

    std::vector<vec3> vertices;
    std::vector<face5> faces;
    Material* materialRefl;

    Dodecahedron(const std::vector<vec3>& _vertices, const std::vector<face5>& _faces, Material* _material) {
        vertices = _vertices;
        faces = _faces;
        material = _material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        for (const face5& face: faces) {
            vec3 p1 = vertices[face.i];
            vec3 p2 = vertices[face.j];
            vec3 p3 = vertices[face.k];
            vec3 u = p2 - p1;
            vec3 v = p3 - p1;
            vec3 normal = normalize(cross(u, v));
            float t = dot(normal,(p1 - ray.start)) / dot(normal, ray.dir);

            if ((t > 0 && hit.t == -1) || (t > 0 && t < hit.t)) { //find the plane that the ray intersects first
                vec3 p = ray.start + ray.dir * t;
                vec3 A = vertices[face.i];
                vec3 B = vertices[face.j];
                vec3 C = vertices[face.k];
                vec3 D = vertices[face.l];
                vec3 E = vertices[face.m];

                bool b1 = isInside(normal,p,A,B);
                bool b2 = isInside(normal,p,B,C);
                bool b3 = isInside(normal,p,C,D);
                bool b4 = isInside(normal,p,D,E);
                bool b5 = isInside(normal,p,E,A);

                if (b1 && b2 && b3 && b4 && b5) {
                    hit.t = t;
                    hit.position = p;
                    hit.normal = normal;
                    hit.material = material;
                }
            }
        }
        return hit;
    }
};

struct Camera {
    vec3 eye, lookat, right, up;
    float fov;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
        eye = _eye;
        lookat = _lookat;
        fov = _fov;
        vec3 w = eye - lookat;
        float focus = length(w);
        right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }

    Ray getRay(int X, int Y) {
        vec3 dir = lookat + right * (2.0 * (X + 0.5) / windowWidth - 1) + up * (2.0 * (Y + 0.5) / windowHeight - 1) - eye;
        return Ray(eye, dir);
    }

    void Animate(float dt) {
        eye = vec3((eye.x - lookat.x) * cosf(dt) + (eye.z - lookat.z) * sinf(dt) + lookat.x,
                   eye.y,
                   -(eye.x - lookat.x) * sinf(dt) + (eye.z - lookat.z) * cosf(dt) + lookat.z);
        set(eye, lookat, up, fov);
    }

};

struct Light {
    vec3 position;
    vec3 Le;
    Light(vec3 _position = vec3(), vec3 _Le = vec3(1, 1, 1)) {
        position = normalize(_position);
        Le = _Le;
    }
    vec3 radiance(float distance) {
        return Le / (powf(distance, 2));
    }
};

const float epsilon = 0.0001f;
const int maxdepth = 5;

class Scene {
    std::vector<Intersectable*> objects;
    std::vector<Light*> lights;
    Camera camera;
    vec3 La;
public:

    void build() {
        vec3 eye = vec3(-0.75, -0.8, 0.5), vup = vec3(0, 0, 1), lookat = vec3(0.5, 0.5, 0.5);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);
        //camera.Animate(0.885398);

        La = vec3(0.2f,0.2f,0.2f);
        Light *redLight = new Light(vec3(0,0.5,0),vec3(1,0,0));
        Light *greenLight = new Light(vec3(0.5,0.5,0),vec3(0,1,0));
        Light *blueLight = new Light(vec3(-0.5,0.5,0),vec3(0,0,1));
        //lights.push_back(redLight);
        //lights.push_back(greenLight);
        //lights.push_back(blueLight);

        //init cube
        vec3 cubeVertexArr[] = {vec3(0,0,0), vec3(0,0,1), vec3(0,1,0), vec3(0,1,1),
                                vec3(1,0,0), vec3(1,0,1), vec3(1,1,0), vec3(1,1,1)};
        std::vector<vec3> cubeVertices(cubeVertexArr,cubeVertexArr + 8);

        Cube::face4 cubeFaceArr[] = {Cube::face4(1,2,6,5),Cube::face4(3,4,2,1),
                                     Cube::face4(7,8,4,3), Cube::face4(5,6,8,7),
                                     Cube::face4(2,4,8,6), Cube::face4(1,3,7,5)};
        std::vector<Cube::face4> cubeFaces(cubeFaceArr, cubeFaceArr + 6);
        Material *refractiveAndRoughMaterial = new RefractiveAndRoughMaterial(vec3(1,1,1),vec3(1,1,1),vec3(0.5, 0.5, 0.5), vec3(0.1, 0.1, 0.1), 100);
        objects.push_back(new Cube(cubeVertices,cubeFaces,refractiveAndRoughMaterial));

        //init dodecahedron
        vec3 vertexArr[] = {vec3(0,0.618,1.618), vec3(0,-0.618,1.618),	vec3(0,-0.618,-1.618),	vec3(0,0.618,-1.618),
                            vec3(1.618,0,0.618), vec3(-1.618,0,0.618),	vec3(-1.618,0,-0.618),	vec3(1.618,0,-0.618),
                            vec3(0.618,1.618,0), vec3(-0.618,1.618,0),	vec3(-0.618,-1.618,0),	vec3(0.618,-1.618,0),
                            vec3(1,1,1),		 vec3(-1,1,1),			vec3(-1,-1,1),			vec3(1,-1,1),
                            vec3(1,-1,-1),		 vec3(1,1,-1),			vec3(-1,1,-1),			vec3(-1,-1,-1) };
        std::vector<vec3> vertices(vertexArr, vertexArr + 20);

        Dodecahedron::face5 faceArr[] = {Dodecahedron::face5(1,2,16,5,13),		Dodecahedron::face5(1,13,9,10,14),
                                         Dodecahedron::face5(1,14,6,15,2),		Dodecahedron::face5(2,15,11,12,16),
                                         Dodecahedron::face5(3,4,18,8,17),		Dodecahedron::face5(3,17,12,11,20),
                                         Dodecahedron::face5(3,20,7,19,4),		Dodecahedron::face5(19,10,9,18,4),
                                         Dodecahedron::face5(16,12,17,8,5),		Dodecahedron::face5(5,8,18,9,13),
                                         Dodecahedron::face5(14,10,19,7,6),		Dodecahedron::face5(6,7,20,11,15) };
        std::vector<Dodecahedron::face5> faces(faceArr, faceArr + 12);

        Material* ceramicPink = new RoughMaterial(vec3(0.3, 0.25, 0.3), vec3(2, 2, 2), 1000);
        //objects.push_back(new Dodecahedron(vertices, faces, ceramicPink));
    }

    void render(std::vector<vec4>& image) {
        for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
            for (int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera.getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }

    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        for (Intersectable * object : objects) {
            Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    vec3 reflect(vec3 V, vec3 N) {
        return V - N * dot(N, V) * 2;
    };

    vec3 refract(vec3 V, vec3 N, float ns){
        float cosa = -dot(V,N);
        float disc = 1 - (1-cosa*cosa) / ns / ns; //n scalar
        if (disc < 0) return vec3(0,0,0);
        return V/ns + N * (cosa/ns - sqrt(disc));
    }

    vec3 Fresnel(vec3 V, vec3 N, vec3 F0) {
        float cosa = -dot(V, N);
        return F0 + (vec3(1, 1, 1) - F0) * powf(1 - cosa, 5);
    }

    vec3 trace(Ray ray, int d = 0) {
        if (d > maxdepth) return La;
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return La; //La
        vec3 outRadiance = vec3(0, 0, 0);

        if (hit.material->rough) {
            outRadiance = La * (1 + fabs(dot(hit.normal, ray.dir))); //L = 0.2 * (1 + dot(N, V) | 0.2 <= L =< 0.4
            for (Light* light : lights) {
                vec3 direction = normalize(light->position - hit.position);
                float distance = length(light->position - hit.position);
                vec3 Le = light->radiance(distance);
                Ray shadowRay(hit.position + hit.normal * epsilon, direction);
                Hit shadowHit = firstIntersect(shadowRay);
                float cosTheta = dot(hit.normal, direction);
                if (cosTheta > 0 && (shadowHit.t < 0 || shadowHit.t > distance)) {
                    outRadiance = outRadiance + Le * hit.material->kd * cosTheta;
                    vec3 halfway = normalize(-ray.dir + direction);
                    float cosDelta = dot(hit.normal, halfway);
                    if (cosDelta > 0) outRadiance = outRadiance + Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
                }
            }
        }

        if (hit.material->reflective) {
            vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
            vec3 reflectedPoint = hit.position + hit.normal * epsilon;
            Ray reflectRay(reflectedPoint,reflectedDir);
            float cosa = -dot(ray.dir, hit.normal);
            vec3 one(1, 1, 1);
            vec3 F = hit.material->F0 + (one - hit.material->F0) * powf(1 - cosa, 5);
            outRadiance = outRadiance + F * trace(reflectRay, d + 1);
        }

        if(hit.material->refractive) {
            float ior = (ray.out) ? hit.material->n.x : (1/hit.material->n.x);
            vec3 refractionDir = refract(ray.dir,hit.normal,ior);
            if (length(refractionDir) > 0) {
                vec3 refractedPoint = hit.position - hit.normal * epsilon;
                Ray refractRay(refractedPoint,refractionDir,!ray.out);
                float cosa = -dot(ray.dir, hit.normal);
                vec3 one(1, 1, 1);
                vec3 F = hit.material->F0 + (one - hit.material->F0) * powf(1 - cosa, 5);
                outRadiance = outRadiance + (vec3(1,1,1) - F) * trace(refractRay, d + 1);
            }

        }

        return outRadiance;
    }

};

GPUProgram gpuProgram;
Scene scene;

const char* vertexSource = R"(
	#version 330
    precision highp float;
	layout(location = 0) in vec2 cVertexPosition;
	out vec2 texcoord;
	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);
	}
)";

const char* fragmentSource = R"(
	#version 330
    precision highp float;
	uniform sampler2D textureUnit;
	in  vec2 texcoord;
	out vec4 fragmentColor;
	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

class FullScreenTexturedQuad {
    unsigned int vao;
    Texture texture;
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
            : texture(windowWidth, windowHeight, image) {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        unsigned int vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    void Draw() {
        glBindVertexArray(vao);
        gpuProgram.setUniform(texture, "textureUnit");
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();
    std::vector<vec4> image(windowWidth * windowHeight);
    long timeStart = glutGet(GLUT_ELAPSED_TIME);
    scene.render(image);
    long timeEnd = glutGet(GLUT_ELAPSED_TIME);
    printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));
    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
    glClearColor(1.0f, 0.5f, 0.8f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    std::vector<vec4> image(windowWidth * windowHeight);
    scene.render(image);
    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {}
void onKeyboardUp(unsigned char key, int pX, int pY) {}
void onMouse(int button, int state, int pX, int pY) {}
void onMouseMotion(int pX, int pY) {}
void onIdle() {}