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

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct Material {
	vec3 ka, kd, ks;		
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

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {		
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
	vec3 position, normal;		
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
	virtual Hit intersect(const Ray& ray) = 0;	
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
		if (hit.t > 0) {
			hit.normal = n;
			hit.material = material;
			return hit;
		}
		return hit;
	}
};

struct Side : public Intersectable {

	Plane* p;		
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
		Hit hit = p->intersect(ray); 
	
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
		Material* _rough = new RoughMaterial(vec3(0.3f,0.2f,0.1f), vec3(2,2,2), 25);
		Material* _portal = new PortalMaterial();

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
			Hit hit = s->intersect(ray); 
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
				bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0)
			bestHit.normal = bestHit.normal * (-1);

		return bestHit;
	}

};

struct MySphere : public Intersectable{
	float a, b, c;
	mat4 Q;

	MySphere(float _a, float _b, float _c, Material* _material) : a(_a), b(_b), c(_c) {
		material = _material;
		Q = mat4(
			(a-1.0f), 0.0f, 0.0f, 0.0f,
			0.0f, (b-1.0f), 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, (c/2.0f),
			0.0f, 0.0f, (c/2.0f), -0.09f
		);
	}

	vec3 gradf(vec4 r) { 
		vec4 g = r * Q * 2;
		return normalize(vec3(g.x, g.y, g.z));
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		vec4 S(ray.start.x, ray.start.y, ray.start.z, 1);
		vec4 D(ray.dir.x, ray.dir.y, ray.dir.z, 0);

		float a = dot(D * Q, D);
		float b = dot(D * Q, S) + dot(S * Q, D);
		float c = dot(S * Q, S);
		float discriminant = b * b - 4.0f * a * c;

		if (discriminant < 0) return hit;

		float t1 = (-b + sqrtf(discriminant)) / 2.0f / a;
		float t2 = (-b - sqrtf(discriminant)) / 2.0f / a;

		if (t1 <= 0) return hit;

		bool _t1 = false;
		bool _t2 = false;

		if (_t1 && _t2) return hit;
		else if (_t1) hit.t = t2;
		else if (_t2) hit.t = t1;
		else hit.t = (t2 > 0) ? t2 : t1;

		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1));
		hit.material = material;
		return hit;
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
	vec3 dir;		
	vec3 Le;		
	Light(vec3 _dir, vec3 _Le) {
		dir = normalize(_dir);
		Le = _Le;
	}
};

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	Camera camera;
	vec3 ambientLight, Le, lightPos;
public:
	void build() {
		vec3 eye = vec3(1,0,0.9), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);

		float fov = 60 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		ambientLight = vec3(0.5f,0.6f,0.6f);
		Le = vec3(0.3f,0.3f,0.5f);
		lightPos = vec3(0.5f, 0.5f, 0.5f);

		objects.push_back(new MySphere(2.1f, 1.7f, 0.1f, new ReflectiveMaterial(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f))));
		objects.push_back(new Dodecahedron());
	}

	// source: https://www.geeks3d.com/20141201/how-to-rotate-a-vertex-by-a-quaternion-in-glsl/
	vec3 rotate(vec3 position, vec3 axis, float angle) {		
		vec4 qr = quatFromAxis(axis, angle);
		vec4 qr_conj = conjugateQuaternion(qr);
		vec4 q_pos = vec4(position.x, position.y, position.z, 0);

		vec4 q_tmp = multiplyQuaternion(qr, q_pos);
		qr = multiplyQuaternion(q_tmp, qr_conj);

		return vec3(qr.x, qr.y, qr.z);
	}

	vec4 quatFromAxis(vec3 axis, float angle)
	{
		vec4 qr;
		float half_angle = (angle * 0.5) * 3.14159 / 180.0;
		qr.x = axis.x * sin(half_angle);
		qr.y = axis.y * sin(half_angle);
		qr.z = axis.z * sin(half_angle);
		qr.w = cos(half_angle);
		return qr;
	}

	vec4 conjugateQuaternion(vec4 q) {
		return vec4(-q.x, -q.y, -q.z, q.w);
	}

	vec4 multiplyQuaternion(vec4 q1, vec4 q2) {
		vec4 qr;
		qr.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
		qr.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
		qr.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
		qr.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
		return qr;
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
			Hit hit = object->intersect(ray); 
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) return ambientLight;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return ambientLight;
		
		vec3 outRadiance(0, 0, 0);
		
		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * ambientLight;	
			vec3 lightDirection = normalize(lightPos - hit.position);

			float costheta = dot(hit.normal, lightDirection);
			if (costheta > 0 ) {	
				outRadiance = outRadiance + Le * hit.material->kd * costheta;
				vec3 halfway = normalize(-ray.dir + lightDirection);
				float cosdelta = dot(hit.normal, halfway);
				if (cosdelta > 0) 
					outRadiance = outRadiance + Le * hit.material->ks * powf(cosdelta, hit.material->shininess);
			}
		}

		if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}

		if (hit.material->type == PORTAL) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;

			ray.start = rotate(hit.position + hit.normal * epsilon, hit.normal, 72);
			ray.dir = rotate(reflectedDir, hit.normal, 72);

			outRadiance = outRadiance + trace(Ray(ray.start + hit.normal * epsilon, ray.dir), depth + 1);
		}

		return outRadiance;
	}
};

GPUProgram gpuProgram; 
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

	void Load(std::vector<vec4>& newImage) {//újra textúrázza az imaget
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

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {

	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);

	fullScreenTexturedQuad->Load(image);
	fullScreenTexturedQuad->Draw();

	glutSwapBuffers();	
								
}

void onKeyboard(unsigned char key, int pX, int pY) {
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY){
}

void onIdle() {
	scene.Animate(0.08f);
	glutPostRedisplay();
}