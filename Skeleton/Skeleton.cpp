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
// Nev    : Pelyhe Adam
// Neptun : U0X77G
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

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";


enum MaterialType { ROUGH, REFLECTIVE, PORTAL};

struct Material {
	vec3 ka, kd, ks;		// ambiens, diff�z, spekul�ris visszaver�k�pess�g
	float shininess;
	vec3 F0;
	MaterialType type;
	Material(MaterialType _type) {
		type = _type;
	}
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd; 
		ks = _ks;
		shininess = _shininess;
	}
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {		// n: t�r�smutat�, kappa: kiolt�si t�nyez�
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) /
			((n + one) * (n + one) + kappa * kappa);
	}
};

struct PortalMaterial : Material{
	PortalMaterial() : Material(PORTAL) {}
};

struct Hit {
	float t;	
	vec3 position, normal;		// position: visszaadja a pontot ahol t�rt�nt �tk�z�s a testtel, normal: fel�let norm�lvektora
	Material* material;
	Hit() {
		t = -1;
	}
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) {
		start = _start;
		dir = normalize(_dir);
	}
};

class Intersectable {
protected: 
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;		// kap egy sugarat �s eld�nti, hogy t�rt�nt-e metsz�s (hit struktura)
};

mat4 TMatrix4(mat4 m) {
	return mat4(
		m.rows[0].x, m.rows[1].x, m.rows[2].x, m.rows[3].x,
		m.rows[0].y, m.rows[1].y, m.rows[2].y, m.rows[3].y,
		m.rows[0].z, m.rows[1].z, m.rows[2].z, m.rows[3].z,
		m.rows[0].w, m.rows[1].w, m.rows[2].w, m.rows[3].w
	);
}

struct Quadric : public Intersectable {
	mat4 Q; //can be symmetric
	std::vector<Quadric*> intersectors; //if positive to an r, then there is no intersection there

	float f(vec4 r) { //r.w = 1
		return dot(r * Q, r); // = 0 for points on the surface
	}

	virtual vec3 gradf(vec4 r) { //r.w = 1
		vec4 g = r * Q * 2;
		return normalize(vec3(g.x, g.y, g.z));
	}

	void Translate(vec3 t) {
		mat4 m = TranslateMatrix(-1 * t); //inverse matrix of the transformation
		Q = m * Q * TMatrix4(m);
	}

	void Translate(float x, float y, float z) {
		Translate(vec3(x, y, z));
	}

	void Scale(vec3 s) {
		mat4 m = ScaleMatrix(vec3(1 / s.x, 1 / s.y, 1 / s.z)); //inverse matrix of the transformation
		Q = m * Q * TMatrix4(m);
	}

	void Scale(float x, float y, float z) {
		Scale(vec3(x, y, z));
	}

	void Rotate(float angle, vec3 w) {
		mat4 m = RotationMatrix(-1 * angle, w); //inverse matrix of the transformation
		Q = m * Q * TMatrix4(m);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		vec4 S(ray.start.x, ray.start.y, ray.start.z, 1);
		vec4 D(ray.dir.x, ray.dir.y, ray.dir.z, 0);

		float a = dot(D * Q, D);
		float b = dot(D * Q, S) + dot(S * Q, D);
		float c = dot(S * Q, S);
		float discr = b * b - 4.0f * a * c;

		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;    // t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		if (t1 <= 0) return hit;

		bool t1CutOff = false;
		bool t2CutOff = false;
		for (auto cutter : intersectors) {
			vec3 t1Pos(ray.start + ray.dir * t1);
			vec4 t1HomPos(t1Pos.x, t1Pos.y, t1Pos.z, 1);
			if (cutter->f(t1HomPos) > 0) t1CutOff = true;

			vec3 t2Pos(ray.start + ray.dir * t2);
			vec4 t2HomPos(t2Pos.x, t2Pos.y, t2Pos.z, 1);
			if (cutter->f(t2HomPos) > 0) t2CutOff = true;
		}
		if (t1CutOff && t2CutOff) return hit;
		else if (t1CutOff) hit.t = t2;
		else if (t2CutOff) hit.t = t1;
		else hit.t = (t2 > 0) ? t2 : t1;

		hit.position = ray.start + ray.dir * hit.t; //+ translate;
		hit.normal = gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1));
		hit.material = material;
		return hit;
	}
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;    // t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};

struct Edge {
	vec3 x1, x2;

	Edge(vec3 _x1, vec3 _x2) {
		x1 = _x1;
		x2 = _x2;
	}

	float distanceFromPoint(vec3 p) {
		float num = length(cross((x2 - x1), (x1 - p)));
		float denom = length(x2 - x1);
		return num / denom;
	}
};

struct Plane : public Intersectable {
	vec3 p;
	vec3 n;

	Plane(vec3 _p, vec3 _n, Material* _material) {
		material = _material;
		n = -normalize(_n);
		p = _p;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		hit.t = dot(n, p - ray.start) / dot(n, ray.dir);
		if (hit.t <= 0) return hit;
		hit.normal = n;
		hit.material = material;
		return hit;
	}
};

struct Side : public Intersectable {

	//vec3 vert[5];
	Plane* p;		// a dodekaeder intersectje meghivja az osszes oldal intersectjet!!!
	Edge* edges[5];
	Material* rough;
	Material* portal;

	Side(std::vector<vec3> v, Material* _portal, Material* _rough) {
		
		edges[0] = new Edge(v[0], v[1]);
		edges[1] = new Edge(v[1], v[2]);
		edges[2] = new Edge(v[2], v[3]);
		edges[3] = new Edge(v[3], v[4]);
		edges[4] = new Edge(v[4], v[0]);

		portal = _portal;
		rough = _rough;
		vec3 n = cross(v[1] - v[0], v[2] - v[0]);
		p = new Plane(v[0], n, portal);
	}

	Hit intersect(const Ray& ray) {
		Hit hit = p->intersect(ray); //  hit.t < 0 if no intersection
	
		if (hit.t > 0) {
			hit.position = ray.start + ray.dir * hit.t;
			for (int i = 0; i < 5; i++) {
				if (edges[i]->distanceFromPoint(hit.position) < 0.1f) {
					hit.material = rough;
					return hit;
				}
			}
		}

		hit.normal = p->n;
		return hit;

	}	
};

struct Dodecahedron : public Intersectable {

	std::vector<Side*> sides;
	std::vector<vec3> vertices;
	std::vector< std::vector<int> > indexes;

	void initializeVertexes() {
		float g = 0.618;
		float G = 1.618;

		vertices.push_back(vec3(0, g, G));
		vertices.push_back(vec3(0, -g, G));
		vertices.push_back(vec3(0, -g, -G));
		vertices.push_back(vec3(0, g, -G));
		vertices.push_back(vec3(G, 0, g));
		vertices.push_back(vec3(-G, 0, g));
		vertices.push_back(vec3(-G, 0, -g));
		vertices.push_back(vec3(G, 0, -g));
		vertices.push_back(vec3(g, G, 0));
		vertices.push_back(vec3(-g, G, 0));
		vertices.push_back(vec3(-g, -G, 0));
		vertices.push_back(vec3(g, -G, 0));
		vertices.push_back(vec3(1, 1, 1));
		vertices.push_back(vec3(-1, 1, 1));
		vertices.push_back(vec3(-1, -1, 1));
		vertices.push_back(vec3(1, -1, 1));
		vertices.push_back(vec3(1, -1, -1));
		vertices.push_back(vec3(1, 1, -1));
		vertices.push_back(vec3(-1, 1, -1));
		vertices.push_back(vec3(-1, -1, -1));

	}

	void initializeIndexes() {
		std::vector<int> tmp;

		tmp.push_back(1);
		tmp.push_back(2);
		tmp.push_back(16);
		tmp.push_back(5);
		tmp.push_back(13);
		indexes.push_back(tmp);
		tmp.clear();

		tmp.push_back(1);
		tmp.push_back(13);
		tmp.push_back(9);
		tmp.push_back(10);
		tmp.push_back(14);
		indexes.push_back(tmp);
		tmp.clear();

		tmp.push_back(1);
		tmp.push_back(14);
		tmp.push_back(6);
		tmp.push_back(15);
		tmp.push_back(2);
		indexes.push_back(tmp);
		tmp.clear();

		tmp.push_back(2);
		tmp.push_back(15);
		tmp.push_back(11);
		tmp.push_back(12);
		tmp.push_back(16);
		indexes.push_back(tmp);
		tmp.clear();

		tmp.push_back(3);
		tmp.push_back(4);
		tmp.push_back(18);
		tmp.push_back(8);
		tmp.push_back(17);
		indexes.push_back(tmp);
		tmp.clear();

		tmp.push_back(3);
		tmp.push_back(17);
		tmp.push_back(12);
		tmp.push_back(11);
		tmp.push_back(20);
		indexes.push_back(tmp);
		tmp.clear();

		tmp.push_back(3);
		tmp.push_back(20);
		tmp.push_back(7);
		tmp.push_back(19);
		tmp.push_back(4);
		indexes.push_back(tmp);
		tmp.clear();

		tmp.push_back(19);
		tmp.push_back(10);
		tmp.push_back(9);
		tmp.push_back(18);
		tmp.push_back(4);
		indexes.push_back(tmp);
		tmp.clear();

		tmp.push_back(16);
		tmp.push_back(12);
		tmp.push_back(17);
		tmp.push_back(8);
		tmp.push_back(5);
		indexes.push_back(tmp);
		tmp.clear();

		tmp.push_back(5);
		tmp.push_back(8);
		tmp.push_back(18);
		tmp.push_back(9);
		tmp.push_back(13);
		indexes.push_back(tmp);
		tmp.clear();

		tmp.push_back(14);
		tmp.push_back(10);
		tmp.push_back(19);
		tmp.push_back(7);
		tmp.push_back(6);
		indexes.push_back(tmp);
		tmp.clear();

		tmp.push_back(6);
		tmp.push_back(7);
		tmp.push_back(20);
		tmp.push_back(11);
		tmp.push_back(15);
		indexes.push_back(tmp);
	}

	void initializeSides() {
		Material* _rough = new RoughMaterial(vec3(0.1f,0.1f,0.1f), vec3(2,2,2), 50);
		Material* _portal = new PortalMaterial();
		//Material* _portal = new ReflectiveMaterial(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f));

		for (auto index : indexes) {
			std::vector<vec3> tmp;
			tmp.push_back(vertices[index[0] - 1]);
			tmp.push_back(vertices[index[1] - 1]);
			tmp.push_back(vertices[index[2] - 1]);
			tmp.push_back(vertices[index[3] - 1]);
			tmp.push_back(vertices[index[4] - 1]);

			Side* s = new Side(tmp, _portal, _rough);
			sides.push_back(s);
		}
	}

	Dodecahedron() {
		initializeVertexes();
		initializeIndexes();
		initializeSides();
	}

	Hit intersect(const Ray& ray) {
		Hit bestHit;
		for (Side* s : sides) {
			Hit hit = s->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
				bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0)
			bestHit.normal = bestHit.normal * (-1);

		return bestHit;
	}

};

struct MySphere : public Quadric {
	float a, b, c;

	MySphere(float _a, float _b, float _c, Material* _material) : a(_a), b(_b), c(_c) {
		material = _material;
		Q = mat4(
			(a-1.0f), 0.0f, 0.0f, 0.0f,
			0.0f, (b-1.0f), 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, (c/2.0f),
			0.0f, 0.0f, (c/2.0f), -0.09f
		);
	}
};

class Camera {
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
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sinf(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sinf(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);

		set(eye, lookat, up, fov);
	}
};

struct Light {
	vec3 dir;		// f�nyforr�s ir�nya
	vec3 Le;		// spektrum
	Light(vec3 _dir, vec3 _Le) {
		dir = normalize(_dir);
		Le = _Le;
	}
};

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 ambientLight;
public:
	void build() {
		vec3 eye = vec3(1, 0, 0.9), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		ambientLight = vec3(0.6f,0.6f,0.6f);
		vec3 lightDirection(1,1,1), Le(3, 3, 3);
		lights.push_back(new Light(lightDirection, Le));

		objects.push_back(new MySphere(2.1f, 1.7f, 0.1f, new ReflectiveMaterial(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f))));
		objects.push_back(new Dodecahedron());
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	void Animate(float dt) {
		camera.Animate(dt);
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 reflect(vec3 inDir, vec3 normal) {
		return inDir - normal * dot(inDir, normal) * 2.0f;
	}

	vec3 frenel(vec3 F0, float cosa) {
		vec3 one(1, 1, 1);
		return F0 + (one - F0) * pow(1 - cosa, 5);
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) return ambientLight;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return ambientLight;
		vec3 outRadiance(0, 0, 0);

		if (hit.material->type == ROUGH) {
			for (Light* light : lights) {
				outRadiance = hit.material->ks * dot(hit.normal, light->dir);
				Ray shadowRay(hit.position + hit.normal * epsilon, light->dir);
				float cosTheta = dot(hit.normal, light->dir);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light->dir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0)
						outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}

			return outRadiance;
		}
		else if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}
		else if (hit.material->type == PORTAL) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1);
		}

		return outRadiance;
	}


};

GPUProgram gpuProgram; // vertex and fragment shaders
//unsigned int vao;	   // virtual world on the GPU
Scene scene;

class FullScreenTexturedQuad {
	unsigned int vao = 0;	// vertex array object id and texture id
	Texture texture;
	unsigned int newTexture = 0;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image) {
		glGenVertexArrays(1, &vao);    // create 1 vertex array object
		glBindVertexArray(vao);        // make it active

		unsigned int vbo;        // vertex buffer objects
		glGenBuffers(1, &vbo);    // Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };    // two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);       // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		glGenTextures(1, &newTexture);
		glBindTexture(GL_TEXTURE_2D, newTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void Load(std::vector<vec4>& newImage) {//�jra text�r�zza az imaget
		glBindTexture(GL_TEXTURE_2D, newTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &newImage[0]);
	}

	void Draw() {
		glBindVertexArray(vao);    // make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, newTexture);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);    // draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {

	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);

	fullScreenTexturedQuad->Load(image);
	fullScreenTexturedQuad->Draw();

	glutSwapBuffers();	// exchange the two buffers
								
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	scene.Animate(0.1f);
	glutPostRedisplay();
}