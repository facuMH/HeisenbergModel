/*
 * =====================================================================================
 *
 *       Filename:  gl_mesh.hpp
 *
 *    Description:  encapsulation of vertex array object, vertex buffer objects, and vertex array pointers
 *
 *    NOTE: assumes shader defines attributes position, color, normal and textureCoordinate
 *
 *    assumes we will be updating vertex information frequently, but not index information
 *
 *        Version:  1.0
 *        Created:  06/13/2014 16:22:02
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Pablo Colapinto (), gmail -> wolftype 
 *
 * =====================================================================================
 */

#ifndef  gl_mesh_INC
#define  gl_mesh_INC

#include <vector>

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"

namespace lynda {

struct Mesh {

    //position in world space
    glm::vec3 mPos;
    //orientation in world space
    glm::quat mRot;
    //scale
    float mScale;
    int N;

    vector<GLuint> VAOs;

    vector<glm::vec3> vertices;
    vector<GLushort> indices;

    vector<float> xs;
    vector<glm::mat4> models;
    //ID of Vertex Attribute
    GLuint positionID, xID, modelID; //  , colorID, textureCoordinateID;          //<-- Attribute IDS
    //A buffer ID and elementID
    GLuint bufferID, elementID, xBufferID, modelBufferID;
    //An array object ID
    GLuint arrayID;

    unsigned int sv3() {
        return sizeof(glm::vec3);
    }
    unsigned int sf1() {
        return sizeof(float);
    }
    unsigned int sv4() {
        return sizeof(glm::vec4);
    }
    unsigned int sm4() {
        return sizeof(glm::mat4);
    }

    Mesh() :
            mScale(1), mRot(1, 0, 0, 0), mPos(0, 0, 0) {
    }

    void getAttributes(GLuint shaderID) {
        // Get attribute locations from SHADER (if these attributes do not exist in shader, ID=-1)
        positionID = 0;  //glGetAttribLocation(shaderID, "position");
        //xID = 1;  //glGetUniformLocation(shaderID, "my_x");
        //modelID = 1;  //glGetUniformLocation(shaderID, "model");
    }

    void buffer() {

        /*-----------------------------------------------------------------------------
         *  CREATE THE VERTEX ARRAY OBJECT
         *-----------------------------------------------------------------------------*/
        //GENVERTEXARRAYS(1, &arrayID);
        for (auto vao : VAOs){

            BINDVERTEXARRAY(vao);

            /*-----------------------------------------------------------------------------
            *  CREATE THE VERTEX BUFFER OBJECT
            *-----------------------------------------------------------------------------*/
            // Generate one buffer
            glGenBuffers(1, &bufferID);
            glBindBuffer(GL_ARRAY_BUFFER, bufferID);
            glBufferData(GL_ARRAY_BUFFER, vertices.size() * sv3(),
                    &(vertices[0]), GL_DYNAMIC_DRAW); //<-- Prep for frequent updates
            glEnableVertexAttribArray(positionID);
            glVertexAttribPointer(positionID, 3, GL_FLOAT, GL_FALSE,
                    sv3(), (void*) 0);
            glVertexAttribDivisor(positionID, 0);

            /*-----------------------------------------------------------------------------
            *  CREATE THE ELEMENT ARRAY BUFFER OBJECT
            *-----------------------------------------------------------------------------*/
            glGenBuffers(1, &elementID);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementID);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLushort),
                    &(indices[0]), GL_DYNAMIC_DRAW); //<-- Update infrequently, if ever
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

            /*-----------------------------------------------------------------------------
            *  ENABLE VERTEX ATTRIBUTES
            *-----------------------------------------------------------------------------*/
            // Tell OpenGL how to handle the buffer of data that is already on the GPU
            //                      attrib    num   type     normalize   stride     offset
            /*
            glGenBuffers(1, &xBufferID);
            glBindBuffer(GL_ARRAY_BUFFER, xBufferID);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * N, &(xs[0]),
                    GL_STREAM_DRAW);
            glEnableVertexAttribArray(xID);
            glVertexAttribPointer(xID, 1, GL_FLOAT, GL_FALSE, sizeof(float),
                    (void*) 0);
            glVertexAttribDivisor(xID, 1);

            glGenBuffers(1, &modelBufferID);
            glBindBuffer(GL_ARRAY_BUFFER, modelBufferID);
            glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * N, &(models[0]),
                    GL_DYNAMIC_DRAW);
            glEnableVertexAttribArray(modelID);
            glVertexAttribPointer(modelID, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4),
                    (void*) 0);
            glVertexAttribDivisor(modelID, 1);

            glEnableVertexAttribArray(modelID + 1);
            glVertexAttribPointer(modelID + 1, 4, GL_FLOAT, GL_FALSE,
                    sizeof(glm::mat4), (void*) (sv4()));
            glVertexAttribDivisor(modelID + 1, 1);

            glEnableVertexAttribArray(modelID + 2);
            glVertexAttribPointer(modelID + 2, 4, GL_FLOAT, GL_FALSE,
                    sizeof(glm::mat4), (void*) (sv4() * 2));
            glVertexAttribDivisor(modelID + 2, 1);

            glEnableVertexAttribArray(modelID + 3);
            glVertexAttribPointer(modelID + 3, 4, GL_FLOAT, GL_FALSE,
                    sizeof(glm::mat4), (void*) (sv4() * 3));
            glVertexAttribDivisor(modelID + 3, 1);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            */

            // Unbind Everything (NOTE: unbind the vertex array object first)
            BINDVERTEXARRAY(0);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    void bind() {
        BINDVERTEXARRAY(arrayID);
    }
    void unbind() {
        BINDVERTEXARRAY(0);
    }

    glm::mat4 model() {
        glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(mScale));
        glm::mat4 rotate = glm::toMat4(mRot);
        glm::mat4 translate = glm::translate(glm::mat4(), mPos);

        return translate * rotate * scale;
    }

    void drawArrays(GLuint mode) {
        glDrawArrays(mode, 0, vertices.size());
    }
    void drawElements(GLuint mode) {
        glDrawElements(mode, indices.size(), GL_UNSIGNED_SHORT, 0);
    }

};

} //lynda::

#endif   /* ----- #ifndef gl_mesh_INC  ----- */
