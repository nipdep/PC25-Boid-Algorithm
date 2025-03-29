#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#define _USE_MATH_DEFINES

#define MODULO(a, n) fmod((a), (n)) + (((a) < 0) * (n))
#define INVERSE(theta) fmod((theta) + M_PI, 2 * M_PI)
#define WIDTH 800
#define HEIGHT 450
#define FLOCKSIZE 400
#define FRAMES 60
#define angularVelocity 1
#define MAX_VELOCITY 20
#define MIN_VELOCITY -20
#define SEPARATION_DISTANCE 8
#define COHESION_DISTANCE 20
#define TURNBACK_DISTANCE 100
#define TURN_FACTOR 0.2
#define CENTERING_FACTOR 0.0005
#define AVOID_FACTOR 0.005
#define MATCHING_FACTOR 0.005

typedef struct
{
     float x;
     float y;
} Vector2;

typedef struct Boid
{
     int id;
     Vector2 position;
     Vector2 velocity; // pixels per second
     float rotation;
     int timestep;
} Boid;

struct LocalFlock
{
     bool cohesion_flock_mask[FLOCKSIZE];
     bool separation_flock_mask[FLOCKSIZE];
} typedef LocalFlock;

void printVector2(const char *label, Vector2 v)
{
     printf("%s: (%.2f, %.2f)\n", label, v.x, v.y);
}

void printBoidInfo(Boid *boid)
{
     // Assuming boid->id exists and is valid
     float x_position = boid->position.x;
     float y_position = boid->position.y;
     float x_velocity = boid->velocity.x;
     float y_velocity = boid->velocity.y;

     // Print in concise format
     printf("%d,%d,%.2f,%.2f,%.2f,%.2f\n",
            boid->id, boid->timestep, x_position, y_position, x_velocity, y_velocity);
}

Boid *newBoid(int id, Vector2 origin, Vector2 velocity)
{
     Boid *boid = malloc(sizeof(Boid));
     *boid = (Boid){id, origin, velocity, 0, 0};

     // rotateBoid(boid, rotation);
     printBoidInfo(boid);
     return boid;
}

int GetRandomValue(int min, int max)
{
     return min + rand() % (max - min + 1);
}

float distance(Vector2 v1, Vector2 v2)
{
     Vector2 delta = {fabsf(v1.x - v2.x), fabsf(v1.y - v2.y)};
     return sqrtf(delta.x * delta.x + delta.y * delta.y);
}

float cap_velocity(float velocity)
{
     if (velocity > MAX_VELOCITY)
     {
          velocity = MAX_VELOCITY;
     }
     else if (velocity < MIN_VELOCITY)
     {
          velocity = MIN_VELOCITY;
     }
     return velocity;
}

LocalFlock getLocalFlock(Boid *boid, Boid **flock, int flockSize)
{
     LocalFlock localFlock;

     for (int i = 0; i < flockSize; i++)
     {
          float dist = distance(flock[i]->position, boid->position);
          // check for alignment mask
          if (dist < COHESION_DISTANCE)
          {
               localFlock.cohesion_flock_mask[i] = true;
          }
          // check for separation mask
          if (dist < SEPARATION_DISTANCE)
          {
               localFlock.separation_flock_mask[i] = true;
          }
     }

     return localFlock;
}

float getRotation(Vector2 v1, Vector2 v2)
{
     Vector2 delta = {v1.x - v2.x, v1.y - v2.y};
     return atan2f(-delta.x, delta.y);
}

float inverse(float angle)
{
     float inverse_angle = angle + M_PI;

     // Keep it in the range [-PI, PI] if desired
     if (inverse_angle > M_PI)
          inverse_angle -= 2.0f * M_PI;

     return inverse_angle;
}

float modulo(float value, float modulus)
{
     float result = fmodf(value, modulus);
     if (result < 0)
     {
          result += modulus;
     }
     return result;
}

void rotateBoid(Boid *boid, float x_rotation, float y_rotation)
{
     float angleRadians = sqrtf(x_rotation * x_rotation + y_rotation * y_rotation);
     boid->rotation += angleRadians;
     // boid->rotation = fmodf(boid->rotation, 2 * M_PI);  // Keep in [0, 2π]

     float speed = sqrtf(boid->velocity.x * boid->velocity.x + boid->velocity.y * boid->velocity.y);
     boid->velocity.x = speed * cosf(boid->rotation);
     boid->velocity.y = speed * sinf(boid->rotation);
}

Vector2 getCohesion(Boid *boid, Boid **flock, LocalFlock localFlock)
{
     Vector2 mean = {0, 0};
     int localflocksize = 0;
     for (int i = 0; i < FLOCKSIZE; i++)
     {
          if (localFlock.cohesion_flock_mask[i])
          {
               mean.x += flock[i]->position.x;
               mean.y += flock[i]->position.y;
               localflocksize++;
          }
     }

     mean = (Vector2){mean.x / localflocksize, mean.y / localflocksize};
     return mean; // getRotation(boid->velocity, mean);
}

Vector2 getAlignment(Boid *boid, Boid **flock, LocalFlock localFlock)
{
     Vector2 mean = {0, 0};
     int localflocksize = 0;
     for (int i = 0; i < FLOCKSIZE; i++)
     {
          if (localFlock.cohesion_flock_mask[i])
          {
               mean.x += flock[i]->velocity.x;
               mean.y += flock[i]->velocity.y;
               localflocksize++;
          }
     }

     mean = (Vector2){mean.x / localflocksize, mean.y / localflocksize};
     return mean; // getRotation(boid->velocity, mean);
}

Vector2 getSeparation(Boid *boid, Boid **flock, LocalFlock localFlock)
{
     Vector2 mean = {0, 0};
     int localflocksize = 0;
     for (int i = 0; i < FLOCKSIZE; i++)
     {
          if (localFlock.separation_flock_mask[i])
          {
               mean.x += boid->position.x - flock[i]->position.x;
               mean.y += boid->position.y - flock[i]->position.y;
               localflocksize++;
          }
     }

     mean = (Vector2){mean.x / localflocksize, mean.y / localflocksize};
     return mean; // getRotation(boid->velocity, mean);
}

void updateBoid(Boid *boid, Boid **flock, int flockSize, int timestep)
{
     LocalFlock localFlock = getLocalFlock(boid, flock, flockSize);
     Vector2 cohesion = getCohesion(boid, flock, localFlock);
     Vector2 alignment = getAlignment(boid, flock, localFlock);
     Vector2 separation = getSeparation(boid, flock, localFlock);

     // float rotation = (cohesion + alignment + separation) / 3;
     float velocity = sqrtf(boid->velocity.x * boid->velocity.x + boid->velocity.y * boid->velocity.y);

     if (boid->position.x < TURNBACK_DISTANCE)
     {
          boid->velocity.x += TURN_FACTOR * fabs(boid->velocity.x) + 3;
     }
     else if (boid->position.x > WIDTH - TURNBACK_DISTANCE)
     {
          boid->velocity.x -= TURN_FACTOR * fabs(boid->velocity.x) + 3;
     }
     else
     {
          boid->velocity.x += (cohesion.x - boid->position.x) * CENTERING_FACTOR + (alignment.x - boid->velocity.x) * MATCHING_FACTOR + (separation.x) * AVOID_FACTOR;
     }

     if (boid->position.y < TURNBACK_DISTANCE)
     {
          boid->velocity.y += TURN_FACTOR * fabs(boid->velocity.y) + 3;
     }
     else if (boid->position.y > HEIGHT - TURNBACK_DISTANCE)
     {
          boid->velocity.y -= TURN_FACTOR * fabs(boid->velocity.y) + 3;
     }
     else
     {
          boid->velocity.y += (cohesion.y - boid->position.y) * CENTERING_FACTOR + (alignment.y - boid->velocity.y) * MATCHING_FACTOR + (separation.y) * AVOID_FACTOR;
     }

     // realign velocity
     // boid->velocity.x += (cohesion.x - boid->position.x) * CENTERING_FACTOR + (alignment.x - boid->velocity.x) * MATCHING_FACTOR + (separation.x) * AVOID_FACTOR;
     // boid->velocity.y += (cohesion.y - boid->position.y) * CENTERING_FACTOR + (alignment.y - boid->velocity.y) * MATCHING_FACTOR + (separation.y) * AVOID_FACTOR;

     // rotate near frame boundaries
     boid->velocity.x = cap_velocity(boid->velocity.x);
     boid->velocity.y = cap_velocity(boid->velocity.y);

     // Vector2 newVelocity = {velocity * cos(rotation), velocity * sin(rotation)};
     // boid->velocity = newVelocity;
     boid->position.x += boid->velocity.x;
     boid->position.y += boid->velocity.y;
     boid->timestep = timestep;

     printBoidInfo(boid);
     boid->rotation = 0;
}

int main(void)
{
     Boid *flock[FLOCKSIZE];

     for (int i = 0; i < FLOCKSIZE; i++)
     {
          Vector2 velocity = (Vector2){GetRandomValue(MIN_VELOCITY, MAX_VELOCITY), GetRandomValue(MIN_VELOCITY, MAX_VELOCITY)};
          flock[i] = newBoid(i, (Vector2){GetRandomValue(0, WIDTH), GetRandomValue(0, HEIGHT)}, velocity);
     }

     for (int t = 0; t < FRAMES; t++)
     {
          for (int i = 0; i < FLOCKSIZE; i++)
               updateBoid(flock[i], flock, FLOCKSIZE, t);
     }

     for (int i = 0; i < FLOCKSIZE; i++)
     {
          // free(flock[i]->position);
          free(flock[i]);
     }

     return 0;
}
