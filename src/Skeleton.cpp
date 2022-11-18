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

const float epsilon = 0.0001f;

bool isInside(vec3 normal, vec3 p, vec3 r1, vec3 r2){
    return dot(cross(r2-r1,p-r1),normal) > 0;
}
inline vec3 operator/(const vec3& v1, const vec3& v2) { return vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z); }

struct Material {
    vec3 ka, kd, ks;
    float shininess;
    Material(vec3 _kd, vec3 _ks, float _shininess) {
        ka = _kd * M_PI;
        kd = _kd;
        ks = _ks;
        shininess = _shininess;
    }
};
struct Hit {
    float t;
    vec3 position, normal;
    Material* material;
    Hit() : t(-1) {};
};
struct Ray {
    vec3 start, dir;
    Ray(vec3 _start = vec3(), vec3 _dir = vec3()) : start(_start), dir(normalize(_dir)) {
        start = _start;
        dir = normalize(_dir);
    }
};

class Intersectable {
protected:
    Material *material;
    bool cull;
    mat4 M, Minv;
public:
    Intersectable(Material* _material, bool culling = true) : material(_material), cull(culling) {
        M = Minv =
        {1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1
        };
    };
    virtual Hit intersect(const Ray& ray) = 0;
    inline bool isCulled() {return cull;};
    virtual void transform(vec3 scale, float alpha, vec3 axis, vec3 dir) {
        M = ScaleMatrix(scale) * RotationMatrix(alpha, axis) * TranslateMatrix(dir);
        Minv = TranslateMatrix(-dir) * RotationMatrix(-alpha, axis) * ScaleMatrix(1/scale);
    }
};

struct Cube : public Intersectable {
    struct face4 {
        unsigned int i, j, k, l;
        face4(unsigned int _i = 0, unsigned int _j = 0, unsigned int _k = 0, unsigned int _l = 0)
                : i(_i - 1), j(_j - 1), k(_k - 1), l(_l - 1){}
    };
    std::vector<vec3> vertices = {
        vec3(-1,-1,-1),
        vec3(-1,-1,1),
        vec3(-1,1,-1),
        vec3(-1,1,1),
        vec3(1,-1,-1),
        vec3(1,-1,1),
        vec3(1,1,-1),
        vec3(1,1,1)
    };
    const std::vector<face4> faces = {
        Cube::face4(1,2,6,5),
        Cube::face4(3,4,2,1),
        Cube::face4(7,8,4,3),
        Cube::face4(5,6,8,7),
        Cube::face4(2,4,8,6),
        Cube::face4(5,7,3,1)
    };

    Cube(Material* _material, bool culling = true) : Intersectable(_material, culling) {}
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

            if ((cull || (dot(ray.dir, normal) < 0)) && ((t > 0 && hit.t == -1) || (t > 0 && t < hit.t))) { //find the plane that the ray intersects first (dot(ray.dir, normal) < 0) &&
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
    void transform(vec3 scale, float alpha, vec3 axis, vec3 dir) {
        vec4 w;
        for (auto &v : vertices) {
            w = {v.x, v.y, v.z, 1};
            w = w * Minv;
            v.x = w.x; v.y = w.y; v.z = w.z;
        }
        Intersectable::transform(scale, alpha, axis, dir);
        for (auto &v : vertices) {
            w = {v.x, v.y, v.z, 1};
            w = w * M;
            v.x = w.x; v.y = w.y; v.z = w.z;
        }
    }
};
struct Dodecahedron : public Intersectable{
    struct face5 {
        unsigned int i, j, k, l, m;
        face5(unsigned int _i = 0, unsigned int _j = 0, unsigned int _k = 0, unsigned int _l = 0, unsigned int _m = 0)
                : i(_i - 1), j(_j - 1), k(_k - 1), l(_l - 1), m(_m - 1) {}
    };

    std::vector<vec3> vertices = {
        vec3(-0.57735, -0.57735, 0.57735),
        vec3(0.934172, 0.356822, 0),
        vec3(0.934172, -0.356822, 0),
        vec3(-0.934172, 0.356822, 0),
        vec3(-0.934172, -0.356822, 0),
        vec3(0, 0.934172, 0.356822),
        vec3(0, 0.934172, -0.356822),
        vec3(0.356822, 0, -0.934172),
        vec3(-0.356822, 0, -0.934172),
        vec3(0, -0.934172, -0.356822),
        vec3(0, -0.934172, 0.356822),
        vec3(0.356822, 0, 0.934172),
        vec3(-0.356822, 0, 0.934172),
        vec3(0.57735, 0.57735, -0.57735),
        vec3(0.57735, 0.57735, 0.57735),
        vec3(-0.57735, 0.57735, -0.57735),
        vec3(-0.57735, 0.57735, 0.57735),
        vec3(0.57735, -0.57735, -0.57735),
        vec3(0.57735, -0.57735, 0.57735),
        vec3(-0.57735, -0.57735, -0.57735)
    };
    const std::vector<face5> faces = {
        face5(2, 3, 19, 12, 15),
        face5(2, 14, 8, 18, 3),
        face5(4, 5, 20, 9, 16),
        face5(4, 17, 13, 1, 5),
        face5(4, 16, 7, 6, 17),
        face5(2, 15, 6, 7, 14),
        face5(3, 18, 10, 11, 19),
        face5(5, 1, 11, 10, 20),
        face5(8, 9, 20, 10, 18),
        face5(7, 16, 9, 8, 14),
        face5(6, 15, 12, 13, 17),
        face5(11, 1, 13, 12, 19)
    };

    Dodecahedron(Material* _material, bool culling = true) : Intersectable(_material, culling) {}
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
    void transform(vec3 scale, float alpha, vec3 axis, vec3 dir) {
        vec4 w;
        for (auto &v : vertices) {
            w = {v.x, v.y, v.z, 1};
            w = w * Minv;
            v.x = w.x; v.y = w.y; v.z = w.z;
        }
        Intersectable::transform(scale, alpha, axis, dir);
        for (auto &v : vertices) {
            w = {v.x, v.y, v.z, 1};
            w = w * M;
            v.x = w.x; v.y = w.y; v.z = w.z;
        }
    }
};
struct Icosahedron : public Intersectable {
    struct face3 {
        unsigned int i, j, k;
        face3(unsigned int _i = 0, unsigned int _j = 0, unsigned int _k = 0)
                : i(_i - 1), j(_j - 1), k(_k - 1){}
    };
    std::vector<vec3> vertices = {
        vec3(0, -0.525731, 0.850651),
        vec3(0.850651, 0, 0.525731),
        vec3(0.850651, 0, -0.525731),
        vec3(-0.850651, 0, -0.525731),
        vec3(-0.850651, 0, 0.525731),
        vec3(-0.525731, 0.850651, 0),
        vec3(0.525731, 0.850651, 0),
        vec3(0.525731, -0.850651, 0),
        vec3(-0.525731, -0.850651, 0),
        vec3(0, -0.525731, -0.850651),
        vec3(0, 0.525731, -0.850651),
        vec3(0, 0.525731, 0.850651)
    };
    const std::vector<face3> faces = {
        face3(2,3,7),
        face3(2,8,3),
        face3(4,5,6),
        face3(5,4,9),
        face3(7,6,12),
        face3(6,7,11),
        face3(10,11,3),
        face3(11,10,4),
        face3(8,9,10),
        face3(9,8,1),
        face3(12,1,2),
        face3(1,12,5),
        face3(7,3,11),
        face3(2,7,12),
        face3(4,6,11),
        face3(6,5,12),
        face3(3,8,10),
        face3(8,2,1),
        face3(4,10,9),
        face3(5,9,1)
    };

    Icosahedron(Material* _material, bool culling = true) : Intersectable(_material, culling) {}

    Hit intersect(const Ray& ray) {
        Hit hit;

        for (const face3& face: faces) {
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

                bool b1 = isInside(normal,p,A,B);
                bool b2 = isInside(normal,p,B,C);
                bool b3 = isInside(normal,p,C,A);

                if (b1 && b2 && b3) {
                    hit.t = t;
                    hit.position = p;
                    hit.normal = normal;
                    hit.material = material;
                }
            }
        }
        return hit;
    }
    void transform(vec3 scale, float alpha, vec3 axis, vec3 dir) {
        vec4 w;
        for (auto &v : vertices) {
            w = {v.x, v.y, v.z, 1};
            w = w * Minv;
            v.x = w.x; v.y = w.y; v.z = w.z;
        }
        Intersectable::transform(scale, alpha, axis, dir);
        for (auto &v : vertices) {
            w = {v.x, v.y, v.z, 1};
            w = w * M;
            v.x = w.x; v.y = w.y; v.z = w.z;
        }
    }
};
struct Cone : public Intersectable {
    vec3 p; //tip point
    vec3 n; //cone axis
    float height;
    float cosa;

    Cone(vec3 _p, vec3 _n, float _height, float _cosa, Material* _material, bool culling = true) : Intersectable(_material, culling) {
        p = _p;
        n = _n;
        height = _height;
        cosa = _cosa;
    }
    void set(vec3 _p, vec3 _n) {
        p = _p;
        n = _n;
    }
    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 sp = ray.start - p;
        float dn = dot(ray.dir, n);
        float xn = dot(sp, n);
        float c2 = cosa*cosa;

        float a = dn*dn - c2 * dot(ray.dir, ray.dir);
        float b = 2 * (dn * xn - c2 * dot(ray.dir, sp));
        float c = xn*xn - c2 * dot(sp, sp);

        float discr = b * b - 4 * a * c;
        if (discr < 0) return hit;

        discr = sqrtf(discr);
        float t, t1, t2;
        t1 = (-b + discr) / (2 * a);
        t2 = (-b - discr) / (2 * a);

        t = (t2 > 0) ? t1 : t2;
        if (t < 0)
            return hit;
        vec3 cp = ray.start + t * ray.dir - p;
        float h = dot(cp, n);
        if (h < 0. || h > height) {
            t = t2;
            cp = ray.start + t * ray.dir - p;
            h = dot(cp, n);
            if (h < 0. || h > height)
                return hit;
        }

        vec3 normal = normalize(cp * dot(n, cp) / dot(cp, cp) - n);
        hit.t = t;
        hit.position = ray.start + ray.dir * t;
        vec3 rp = normalize(hit.position - p);
        //hit.normal = hit.position - (length(rp)/cosa * n + p);
        hit.normal = normalize(cross(cross(n, rp), rp));
        hit.material = material;
        return hit;
    }
};

struct Camera {
    vec3 eye, lookat, right, up;
    float fov;

    Camera() {
        vec3 _eye = vec3(1.0, 0, -1.4);
        vec3 _vup = vec3(0, 1, 0);
        vec3 _lookat = vec3(0, 0, 0);
        float _fov = 50 * M_PI / 180;
        set(_eye, _lookat, _vup, _fov);
    }
    Camera(vec3 _eye, vec3 _lookat, vec3 _vup, float _fov) {
        set(_eye, _lookat, _vup, _fov);
    }
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
        eye = _eye;
        lookat = _lookat;
        fov = _fov;
        vec3 w = eye - lookat;
        float focus = length(w);
        right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }
    Ray getRay(int x, int y) {
        vec3 dir = lookat + right * (2.0f*x / windowWidth - 1) + up * (1 - 2.0*y / windowHeight) - eye;
        return Ray(eye, dir);
    }
    void Animate(float dt) {
        eye = vec3((eye.x - lookat.x) * cosf(dt) + (eye.z - lookat.z) * sinf(dt) + lookat.x,
                   eye.y,
                   -(eye.x - lookat.x) * sinf(dt) + (eye.z - lookat.z) * cosf(dt) + lookat.z);
        set(eye, lookat, vec3(0, 1, 0), fov);
    }
};
struct Light {
    vec3 position;
    vec3 Le;
    Cone *cone;
    Light(const vec3 &_position, vec3 _Le = vec3(1, 1, 1)) : position(_position), Le(_Le) {}
    vec3 radiance(float distance) {
        return Le / (powf(distance, 2));
    }
    Light(Cone* _c, const vec3 _position, vec3 _Le = vec3(1, 1, 1)) : position(_position), Le(_Le), cone(_c) {}
    void set(vec3 p) {
        position = p;
    }
};

/// TODO: Please kill me
const vec3 LIGHTPOS(0.0, 0.0, 0.0);

class Scene {
    std::vector<Intersectable*> objects;
    std::vector<Light> lights;
    Camera camera;
    vec3 La;
public:
    void build() {
        La = vec3(0.2f,0.2f,0.2f);
        Material* ceramicPink = new Material(vec3(0.3, 0.25, 0.3), vec3(2, 2, 2), 40);

        Cube* c = new Cube(ceramicPink, false);
        c->transform(vec3(0.5,0.5,0.5), 0, vec3(0, 1, 0), vec3(0, 0, 0));
        objects.push_back(c);
        Dodecahedron* d = new Dodecahedron(ceramicPink);
        d->transform(vec3(0.28,0.28,0.28), 0, vec3(0, 1, 0), vec3(0.23843184, -0.23843184, 0.03843184));
        objects.push_back(d);
        Icosahedron* i = new Icosahedron(ceramicPink);
        i->transform(vec3(0.2,0.2,0.2), M_PI / 2.0, vec3(0, 1, 0), vec3(0, -0.31285678, -0.31285678));
        objects.push_back(i);

        Cone* c1 = new Cone(vec3(-0.3, 0.3, 0.5),normalize(vec3(0,0,-1)),0.08,0.9,ceramicPink);
        Cone* c2 = new Cone(vec3(-0.3, 0.5, 0),normalize(vec3(0,-1,0)),0.08,0.9,ceramicPink);
        Cone* c3 = new Cone(vec3(0.155912, -0.071450, -0.119936),normalize(vec3(0, 0.525731, -0.850651)),0.08,0.9,ceramicPink);
        objects.push_back(c1);
        objects.push_back(c2);
        objects.push_back(c3);

        Light redLight = Light(c1,c1->p + (c1->n * 0.005),vec3(10,0,0));
        Light greenLight = Light(c2,c2->p + (c2->n * 0.005),vec3(0,10,0));
        Light blueLight = Light(c3,c3->p + (c3->n * 0.005),vec3(0,0,10));
        lights.push_back(redLight);
        lights.push_back(greenLight);
        lights.push_back(blueLight);
    }
    void render(std::vector<vec4>& image) {
        for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
            for (int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera.getRay(X, Y));

                image[(windowHeight - Y - 1) * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }
    void moveCone(int x, int y) {
        Ray ray = camera.getRay(x, y);
        Hit hit = firstIntersect(ray);
        Light* closest = nullptr;
        for (auto& l : lights) {
            if (closest == nullptr)
                closest = &l;
            else
                if (length(closest->position - hit.position) > length(l.position - hit.position))
                    closest = &l;
        }
        closest->cone->set(hit.position, hit.normal);
        closest->set(hit.position + hit.normal * 0.005);
    }
    bool shadowIntersect(Ray ray, float max) {
        for (Intersectable * object : objects) {
            Hit hit = object->intersect(ray);
            if (hit.t > 0 && hit.t < max)
                return true;
        }
        return false;
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
    vec3 trace(Ray ray) {
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return vec3(0, 0, 0);

        vec3 outRadiance = La * (1 + dot(hit.normal, -ray.dir)); //L = 0.2 * (1 + dot(N, V) | 0.2 <= L =< 0.4
        vec3 direction;
        for (Light light : lights) {
            direction = light.position - hit.position;
            Ray shadowRay(hit.position + hit.normal * epsilon, direction);
            if (!shadowIntersect(shadowRay, length(direction))) {
                direction = normalize(direction);
                float cosTheta = dot(hit.normal, direction);
                if (cosTheta > 0) {
                    float distance = length(light.position - hit.position);
                    vec3 Le = light.radiance(distance);
                    outRadiance = outRadiance + Le * hit.material->kd * cosTheta;
                    vec3 halfway = normalize(-ray.dir + direction);
                    float cosDelta = dot(hit.normal, halfway);
                    if (cosDelta > 0)
                        outRadiance = outRadiance + Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
                }
            }
        }
        return outRadiance;
    }
    void animate() {
//        camera.Animate(-30.0 * M_PI / 180.0);
    }
};

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

GPUProgram gpuProgram;
Scene scene;
static unsigned int frameRendered = 0;

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
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}
void onDisplay() {
    glClearColor(1.0f, 0.5f, 0.8f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    std::vector<vec4> image(windowWidth * windowHeight);
    scene.render(image);
    delete fullScreenTexturedQuad;
    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();
}
int selector = 0;

void onKeyboard(unsigned char key, int pX, int pY) {}
void onKeyboardUp(unsigned char key, int pX, int pY) {}
void onMouse(int button, int state, int pX, int pY) {
    if (state == GLUT_DOWN){
        scene.moveCone(pX, pY);
        glutPostRedisplay();
    }
}
void onMouseMotion(int pX, int pY) {}
void onIdle() {
//    frameRendered += 1;
//    scene.animate();
//    glutPostRedisplay();
}