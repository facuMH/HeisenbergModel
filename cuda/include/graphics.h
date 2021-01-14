#include <gltools/glfw_app.hpp>
#include <gltools/gl_shader.hpp>
#include <gltools/gl_macros.hpp>
#include <gltools/gl_mesh.hpp>
#include <gltools/gl_error.hpp>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "variables.h"

#define CHOME glm::vec4(41.0f/255.0f, 241.0f/255.0f, 1.0f, 1.0f) // #F1F1FF
#define IRON glm::vec4(238.0f/255.0f, 97.0f/255.0f, 23.0f/255.0f, 1.0f) // #ee6123
#define RED glm::vec4(1.0f, .0f, .0f, 0.50f)
#define GREEN glm::vec4(.0f, 1.0f, .0f, 1.0f)
#define BLUE glm::vec4(.0f, .0f, 1.0f, 0.75f)
#define WHITE glm::vec4(1.0f, 1.0f, 1.0f, 0.25f) //#FFFFFF
#define BLACK glm::vec4(.0f, .0f, .0f, 1.0f) //#FFFFFF

#define ORIGINAL glm::vec3(0,0,2*(L+2))

#define v3 glm::vec3

using namespace lynda;
using namespace std;
/*-----------------------------------------------------------------------------
 *  SHADER CODE
 *-----------------------------------------------------------------------------*/
const char * vert = GLSL(120,
  attribute vec4 position;

  varying vec4 dstColor;

  uniform mat4 model;
  uniform mat4 view;                 //<-- 4x4 Transformation Matrices
  uniform mat4 projection;

  void main(){
      if (position.x == 0.0f){
          dstColor = vec4(1.0f, 1.0f, 1.0f, 0.25f);
      } else {
          dstColor =  position.x > 0.0f ? vec4(1.0f, 0.0f, 0.0f, 0.50f) : vec4(0.0f, 0.0f, 1.0f, 0.75f);
      }
    gl_Position = projection * view * model * position;   //<-- Apply transformation
  }
);

const char * frag = GLSL(120,
  varying vec4 dstColor;
  void main(){
    gl_FragColor = dstColor;
  }
);

/*-----------------------------------------------------------------------------
 *  A Mouse Struct
 *-----------------------------------------------------------------------------*/
struct Mouse{
  bool isDown = false;
  float  lastX, lastY;
  float  dx, dy;
  float  x, y;
};

/*-----------------------------------------------------------------------------
 *  OUR APP
 *-----------------------------------------------------------------------------*/
struct MyApp : public App{

  Shader * shader;
  Mesh mesh;

  //ID of Vertex Attribute
  //GLuint positionID;

  //A buffer ID
  //GLuint bufferID, elementID;                   //<-- add an elementID
  //An array ID
  //GLuint arrayID;

  //ID of Uniforms
  GLuint viewID, projectionID;
  GLuint modelID;

  int L;
  int currentL;
  int change;
  int index, rbx, ixf;
  float x;

  glm::mat4 aux_model;
  vector<v3> new_head;
  //GLint my_x_Location;

  float eyeX,eyeZ,eyeY;
  float radius = 5.0f;
  int keyCode;
  float lastEyePos = 0.0f;
  Mouse mouse;
  glm::vec3 eyePos, tmPos, aux, target;                     //<-- camera position


  MyApp(const int main_L) : App() {  };
  MyApp() = default;

  void init(const int main_L){

    L = main_L;
    currentL = L;
    mesh.N = 8*L*L*L;
    keyCode = -1;
    eyePos = ORIGINAL;
    change = 2*L;

    std::vector<glm::vec3> arrow = {
      { glm::vec3(0.0f, -0.5f, 0.0f)},
      { glm::vec3(0.0f, 0.50f, 0.0f)},
      //{ glm::vec3(0.15f, 0.25f, 0.0f)},
      //{ glm::vec3(-0.15f, 0.25f, 0.0f)}
    };
    mesh.vertices = arrow;


  //       2
  //  0 -----------1
  //       3



    std::vector<GLushort> indexs = {
      0,1,
      1,2,
      2,3,
      3,1
    };
    mesh.indices = indexs;

    for (int i=0; i<mesh.N; i++) {
        //mesh.xs.push_back(-1.0f);
        mesh.VAOs.push_back(0);
        //mesh.models.push_back(glm::mat4(1.0f));
        GENVERTEXARRAYS(1, &mesh.VAOs[i]);

    }
    /*-----------------------------------------------------------------------------
     *  CREATE THE SHADER
     *-----------------------------------------------------------------------------*/
    shader = new Shader( vert, frag );


    // Get uniform locations
    viewID = glGetUniformLocation(shader -> id(), "view");
    projectionID = glGetUniformLocation(shader -> id(), "projection");
    modelID = glGetUniformLocation(shader -> id(), "model");

    // Get attribute locations
    mesh.getAttributes(shader->id());
    mesh.buffer();

  }


  void update_arrows(const float * mx, const float * my, const float * mz,
                     const grid_color color, const int i, const int j, const int k){

      ixf = 2*i + STRF(color,j,k);   // i on full size matrix
      index = IX(ixf, j, k);   // index for full
      rbx = RB(i,j,k);
      x = mx[rbx];

      aux_model = glm::mat4(1.0f);

      aux_model = glm::translate(aux_model, glm::vec3(ixf-currentL,j-currentL,k-currentL));

      if (ixf>(currentL*2) || j>(currentL*2) || k>(currentL*2)) {
          aux_model = glm::scale(aux_model, glm::vec3(0,0,0));
      } else {
          aux_model = glm::scale(aux_model, glm::vec3(0.5f, 0.5f, 0.5f));
      }

      new_head = vector<v3>{ {v3(0.0f)} , {v3(x, my[rbx], mz[rbx])}};

      BINDVERTEXARRAY(mesh.VAOs[index]);
        glUniformMatrix4fv( modelID, 1, GL_FALSE, glm::value_ptr(aux_model) );
        glBindBuffer( GL_ARRAY_BUFFER, mesh.bufferID);
        //glBufferData( GL_ARRAY_BUFFER, mesh.sv3(), mesh.sv3()*2, &(new_head[0]));
        glBufferData(GL_ARRAY_BUFFER, new_head.size() * mesh.sv3(),
                &(new_head[0]), GL_DYNAMIC_DRAW); //<-- Prep for frequent updates

        mesh.drawArrays(GL_LINE_STRIP);
      BINDVERTEXARRAY(0);


      //mesh.models[index] = aux_model;
      //mesh.xs[index] = x;

  }

  void onDraw(float * mx_r, float * my_r, float * mz_r,
              float * mx_b, float * my_b, float * mz_b){
    /*-----------------------------------------------------------------------------
     *  Every frame, check for keyboard input
     *-----------------------------------------------------------------------------*/
    if (keyCode!=-1){
      if (keyCode==GLFW_KEY_UP){
      glm::quat q = glm::angleAxis(-.01f, glm::vec3(1,0,0));
          target += glm::vec3(0,0.1f,0);
      //if (abs(aux[1]) < abs(eyePos[2]))
      //  eyePos = q * eyePos;
          eyePos += glm::vec3(0,0.1f,0);
      }
      if (keyCode==GLFW_KEY_DOWN){
          glm::quat q = glm::angleAxis(.01f, glm::vec3(1,0,0));
      //aux = q * eyePos;
      //if (abs(aux[1]) < abs(eyePos[2]))
      //  eyePos = q * eyePos;
          eyePos += glm::vec3(0,-0.1f,0);
          target += glm::vec3(0,-0.1f,0);
      }
      if (keyCode==GLFW_KEY_LEFT){
          glm::quat q = glm::angleAxis(-.01f, glm::vec3(0,1,0));
          eyePos += glm::vec3(-0.1f,0,0) ;
          target += glm::vec3(-0.1f,0,0) ;
      }
      if (keyCode==GLFW_KEY_RIGHT){
          glm::quat q = glm::angleAxis(.01f, glm::vec3(0,1,0));
          eyePos = eyePos + glm::vec3(0.1f,0,0);
          target+= glm::vec3(0.1f,0,0) ;
      }
      if (keyCode==GLFW_KEY_ENTER){
        std::cout << "press enter to continue."<< "\n";
        cin.get();
      }
      if (keyCode==GLFW_KEY_R){
        eyePos = ORIGINAL;
        target = glm::vec3(0,0,0);
      }
      if (keyCode==GLFW_KEY_C){
        exit(0);
      }
      if (keyCode==GLFW_KEY_1){
          currentL=L;
      }
      if (keyCode==GLFW_KEY_2){
          currentL=L/2;
      }
      if (keyCode==GLFW_KEY_3){
          currentL=L/3;
      }
      if (keyCode==GLFW_KEY_4){
          currentL=L/4;
      }
      if (keyCode==GLFW_KEY_5){
          currentL=L/5;
      }
      if (keyCode==GLFW_KEY_6){
          currentL=L/6;
      }
      if (keyCode==GLFW_KEY_7){
          currentL=L/7;
      }
      if (keyCode==GLFW_KEY_8){
          currentL=L/8;
      }
      if (keyCode==GLFW_KEY_9){
          currentL=L/9;
      }
      if (keyCode==GLFW_KEY_0){
          currentL=L/10;
      }
    }

    glm::mat4 view = glm::lookAt( eyePos, target , glm::vec3(0,1,0) );
    glm::mat4 proj = glm::perspective( fov, ((float)window().width())/window().height(), 0.1f,-10.f);

    //mesh.bind();

    glUniformMatrix4fv( viewID, 1, GL_FALSE, glm::value_ptr(view) );
    glUniformMatrix4fv( projectionID, 1, GL_FALSE, glm::value_ptr(proj) );

    /*-----------------------------------------------------------------------------
     *  Render a whole bunch of spinning triangles
     *-----------------------------------------------------------------------------*/

    for (int k = 0; k < 2*currentL; ++k){
        for (int j = 0; j < 2*currentL; ++j){
            for (int i = 0; i < currentL; ++i){
                update_arrows( mx_r, my_r, mz_r, RED_TILES, i, j, k);
                update_arrows( mx_b, my_b, mz_b, BLACK_TILES, i, j, k);
          }
        }
    }
/*
    glBindBuffer(GL_ARRAY_BUFFER, mesh.xBufferID);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*mesh.N, &(mesh.xs[0]), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, mesh.modelBufferID);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4)*mesh.N, &(mesh.models[0]), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDrawArraysInstanced(GL_LINE_STRIP, 0, 4, mesh.N); // 4 vertices per instance
    mesh.unbind();
*/
  }

    /*-----------------------------------------------------------------------------
   *  Mouse Interactions
   *-----------------------------------------------------------------------------*/
  void onMouseMove(float  x, float  y){
    mouse.x = x;
    mouse.y = y;
    if (mouse.isDown) onMouseDrag(x,y);
  }

  void onMouseDown(int button, int action){
    if(action == GLFW_PRESS){
       mouse.isDown = true;
       double x,y;
       glfwGetCursorPos(window().window, &x, &y);
       mouse.lastX = (float)x;
       mouse.lastY = (float)y;
       tmPos = eyePos;
    } else {
      mouse.isDown = false;
    }
  }

  void onMouseDrag(float x, float y){
    mouse.dx = mouse.x - mouse.lastX;
    mouse.dy = mouse.y - mouse.lastY;

    float wx = mouse.dx / window().width();
    float wy = mouse.dy / window().height();
    glm::vec3 axis = glm::cross( glm::vec3(0,0,1), glm::vec3(-wx,wy,0) );
    glm::quat q = glm::angleAxis( glm::length(axis), glm::normalize(axis) );
    aux = q * tmPos;
    eyePos= q * tmPos;
  }

  void onKeyDown(int key, int action){
      if(action == GLFW_PRESS)
          keyCode = key;
     //cout << "key pressed" << endl;
      if(action == GLFW_RELEASE){
          keyCode = -1;
     //cout << "key released" << endl;
      }
  }


  //app.start(mx, my, mz);
  void drawOne(float * mx_r, float * my_r, float * mz_r,
               float * mx_b, float * my_b, float * mz_b){

    mWindow.setViewport();

    glClearColor(0.0f,0.0f,0.0f,0.0f);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    onDraw(mx_r, my_r, mz_r, mx_b, my_b, mz_b);

    mWindow.swapBuffers();      //<-- SWAP BUFFERS
    glfwPollEvents();           //<-- LISTEN FOR WINDOW EVENTS
  }

};
