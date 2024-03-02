#include "vec.h"
#include "array.h"

bool circumcircle(
    float *xc, float *yc, float *rsqr,
    float xp, float yp,
    float x1, float y1,
    float x2, float y2,
    float x3, float y3,
    float eps = 1e-8f)
{    
    float fabsy1y2 = fabs(y1 - y2);
    float fabsy2y3 = fabs(y2 - y3);
    
    if (fabsy1y2 < eps && fabsy2y3 < eps)
    {
        return false;
    }
    
    if (fabsy1y2 < eps)
    {
        float m2 = - (x3 - x2) / (y3 - y2);
        float mx2 = (x2 + x3) / 2.0f;
        float my2 = (y2 + y3) / 2.0f;
        *xc = (x2 + x1) / 2.0f;
        *yc = m2 * (*xc - mx2) + my2;
    }
    else if (fabsy2y3 < eps)
    {
        float m1 = - (x2 - x1) / (y2 - y1);
        float mx1 = (x1 + x2) / 2.0f;
        float my1 = (y1 + y2) / 2.0f;
        *xc = (x3 + x2) / 2.0f;
        *yc = m1 * (*xc - mx1) + my1;
    }
    else
    {
        float m1 = - (x2 - x1) / (y2 - y1);
        float m2 = - (x3 - x2) / (y3 - y2);
        float mx1 = (x1 + x2) / 2.0f;
        float mx2 = (x2 + x3) / 2.0f;
        float my1 = (y1 + y2) / 2.0f;
        float my2 = (y2 + y3) / 2.0f;
        *xc = (m1 * mx1 - m2 * mx2 + my2 - my1) / (m1 - m2);
        if (fabsy1y2 > fabsy2y3)
        {
            *yc = m1 * (*xc - mx1) + my1;
        }
        else
        {
            *yc = m2 * (*xc - mx2) + my2;
        }
    }
    
    float dx = x2 - *xc;
    float dy = y2 - *yc;
    *rsqr = dx*dx + dy*dy;
    
    float dxp = xp - *xc;
    float dyp = yp - *yc;
    float drsqr = dxp*dxp + dyp*dyp;
    
    return (drsqr - *rsqr) <= eps;
}

struct delauney_point
{
    int index;
    float x, y;
};

struct delauney_tri
{
    int p1, p2, p3;   
};

struct delauney_edge
{
    int p1, p2;
};

int delauney_sort(const void *v1, const void *v2)
{
    delauney_point* p1 = (delauney_point*)v1;
    delauney_point* p2 = (delauney_point*)v2;
    
    if (p1->x < p2->x)
    {
        return -1;
    }
    else if (p1->x > p2->x)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int delauney_triangulate(
    int *ntris, 
    slice1d<delauney_tri> tris, 
    int npoints, 
    slice1d<delauney_point> points)
{
    assert(points.size == npoints + 3);
    assert(tris.size == npoints * 3);

    // Sort Vertices by x coordinate

    qsort(points.data, npoints, sizeof(delauney_point), delauney_sort);

    // Allocate Temp Arrays

    int trimax = 4 * npoints;
    int nedge = 0;

    array1d<bool> complete(trimax);
    array1d<delauney_edge> edges(100);

    // Find the maximum and minimum vertex bounds.
    // This is to allow calculation of the bounding triangle
    float xmin = points(0).x;
    float ymin = points(0).y;
    float xmax = xmin;
    float ymax = ymin;
    for (int i = 1; i < npoints; i++)
    {
        xmin = minf(xmin, points(i).x);
        xmax = maxf(xmax, points(i).x);
        ymin = minf(ymin, points(i).y);
        ymax = maxf(ymax, points(i).y);
    }

    float dx = xmax - xmin;
    float dy = ymax - ymin;
    float dmax = (dx > dy) ? dx : dy;
    float xmid = (xmax + xmin) / 2.0f;
    float ymid = (ymax + ymin) / 2.0f;

    // Set up the supertriangle
    // This is a triangle which encompasses all the sample points.
    // The supertriangle coordinates are added to the end of the
    // vertex list. The supertriangle is the first triangle in
    // the triangle list.
    points(npoints+0).x = xmid - 20 * dmax;
    points(npoints+0).y = ymid - dmax;
    points(npoints+1).x = xmid;
    points(npoints+1).y = ymid + 20 * dmax;
    points(npoints+2).x = xmid + 20 * dmax;
    points(npoints+2).y = ymid - dmax;
    tris(0).p1 = npoints;
    tris(0).p2 = npoints+1;
    tris(0).p3 = npoints+2;
    complete(0) = false;
    *ntris = 1;

    // Include each point one at a time into the existing mesh
    for (int i = 0; i < npoints; i++)
    {
        float xp = points(i).x;
        float yp = points(i).y;
        int nedge = 0;

        // Set up the edge buffer.
        // If the point (xp,yp) lies inside the circumcircle then the
        // three edges of that triangle are added to the edge buffer
        // and that triangle is removed.
        for (int j = 0; j < (*ntris); j++)
        {
            if (complete(j)) { continue; }

            float x1 = points(tris(j).p1).x;
            float y1 = points(tris(j).p1).y;
            float x2 = points(tris(j).p2).x;
            float y2 = points(tris(j).p2).y;
            float x3 = points(tris(j).p3).x;
            float y3 = points(tris(j).p3).y;

            float xc, yc, r;
            bool inside = circumcircle(&xc, &yc, &r, xp, yp, x1, y1, x2, y2, x3, y3);

            if (xc < xp && ((xp-xc)*(xp-xc)) > r)
            {
                complete(j) = true;           
            }

            if (inside)
            {
                // Check that we haven't exceeded the edge list size
                if (nedge+3 >= edges.size)
                {
                    edges.resize(edges.size + 100);
                }
                
                edges(nedge+0).p1 = tris(j).p1;
                edges(nedge+0).p2 = tris(j).p2;
                edges(nedge+1).p1 = tris(j).p2;
                edges(nedge+1).p2 = tris(j).p3;
                edges(nedge+2).p1 = tris(j).p3;
                edges(nedge+2).p2 = tris(j).p1;
                nedge += 3;
                tris(j) = tris((*ntris)-1);
                complete(j) = complete((*ntris)-1);
                (*ntris)--;
                j--;
            }
        }

        // Tag multiple edges
        // Note: if all triangles are specified anticlockwise then all
        // interior edges are opposite pointing in direction.
        for (int j = 0; j < nedge - 1; j++)
        {
            for (int k = j + 1; k < nedge; k++)
            {
                if ((edges(j).p1 == edges(k).p2) && (edges(j).p2 == edges(k).p1))
                {
                    edges(j).p1 = -1;
                    edges(j).p2 = -1;
                    edges(k).p1 = -1;
                    edges(k).p2 = -1;
                }

                // Shouldn't need the following, see note above
                if ((edges(j).p1 == edges(k).p1) && (edges(j).p2 == edges(k).p2))
                {
                    edges(j).p1 = -1;
                    edges(j).p2 = -1;
                    edges(k).p1 = -1;
                    edges(k).p2 = -1;
                }
            }
        }

        // Form new triangles for the current point
        // Skipping over any tagged edges.
        // All edges are arranged in clockwise order.
        for (int j = 0; j < nedge; j++)
        {
            if (edges(j).p1 < 0 || edges(j).p2 < 0)
            {
                continue;
            }

            if ((*ntris) >= trimax)
            {
                return 4;
            }

            tris(*ntris).p1 = edges(j).p1;
            tris(*ntris).p2 = edges(j).p2;
            tris(*ntris).p3 = i;
            complete(*ntris) = false;
            (*ntris)++;
        }
    }

    // Remove triangles with supertriangle vertices
    // These are triangles which have a vertex number greater than npoints
    for (int i = 0; i < (*ntris); i++)
    {
        if (tris(i).p1 >= npoints || tris(i).p2 >= npoints || tris(i).p3 >= npoints)
        {
            tris(i) = tris((*ntris)-1);
            (*ntris)--;
            i--;
        }
    }

    return 0;
}
