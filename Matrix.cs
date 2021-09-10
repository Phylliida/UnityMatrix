using System;
using UnityEngine;

// Modified from https://msdn.microsoft.com/en-us/magazine/mt736457.aspx
public class Matrix
{
    public float[][] mat;
    public int rows
    {
        get { return mat.Length; }
    }
    public int cols
    {
        get { return mat[0].Length; }
    }

    public Matrix(int rows, int cols)
    {
        mat = MatrixCreate(rows, cols);
    }

    public Matrix(float[][] values)
    {
        mat = new float[values.GetLength(0)][];
        for (int i = 0; i < values.GetLength(0); i++)
        {
            mat[i] = new float[values[0].Length];
        }
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                mat[i][j] = values[i][j];
    }


    public Matrix(Matrix4x4 m)
    {
        Matrix res = new Matrix(new float[] {
            m.m00, m.m01, m.m02, m.m03,
            m.m10, m.m11, m.m12, m.m13,
            m.m20, m.m21, m.m22, m.m23,
            m.m30, m.m31, m.m32, m.m33
        }, 4, 4);

        mat = res.mat;
    }
    public Matrix(Transform transform, bool useLocalValues)
    {
        /*
        Matrix4x4 m = transform.localToWorldMatrix;
        Matrix res = new Matrix(new float[] {
            m.m00, m.m01, m.m02, m.m03,
            m.m10, m.m11, m.m12, m.m13,
            m.m20, m.m21, m.m22, m.m23,
            m.m30, m.m31, m.m32, m.m33
        }, 4, 4);

        mat = res.mat;
        */

        if (useLocalValues)
        {
            Matrix rot = new Matrix(Matrix4x4.TRS(transform.localPosition, transform.localRotation, transform.localScale));
            mat = rot.mat;
        }
        else
        {
            Matrix rot = new Matrix(Matrix4x4.TRS(transform.position, transform.rotation, transform.lossyScale));
            mat = rot.mat;
        }
    }

    public Matrix T
    {
        get {
            float[] resValues = new float[rows * cols];

            int pos = 0;
            for (int i = 0; i < cols; ++i)
                for (int j = 0; j < rows; ++j)
                    resValues[pos++] = mat[j][i];
            return new Matrix(resValues, cols, rows);
        }
    }

    public Matrix(Vector4 vec)
    {
        Matrix res = new Matrix(new float[] { vec.x, vec.y, vec.z, vec.w }, 4, 1);
        mat = res.mat;
    }

    public Matrix(Vector3 vec)
    {
        Matrix res = new Matrix(new float[] { vec.x, vec.y, vec.z }, 3, 1);
        mat = res.mat;
    }

    public Matrix(Vector2 vec)
    {
        Matrix res = new Matrix(new float[] { vec.x, vec.y }, 2, 1);
        mat = res.mat;
    }

    // Fro http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
    public Matrix(Quaternion quat)
    {
        mat = MatrixCreate(3, 3);

        Vector4 q = new Vector4(quat.x, quat.y, quat.z, quat.w);
        q = q.normalized;

        float x = q.x;
        float y = q.y;
        float z = q.z;
        float w = q.w;

        mat[0][0] = 1 - 2 * y * y - 2 * z * z;
        mat[0][1] = 2 * x * y - 2 * z * w;
        mat[0][2] = 2 * x * z + 2 * y * w;

        mat[1][0] = 2 * x * y + 2 * z * w;
        mat[1][1] = 1 - 2 * x * x - 2 * z * z;
        mat[1][2] = 2 * y * z - 2 * x * w;

        mat[2][0] = 2 * x * z - 2 * y * w;
        mat[2][1] = 2 * y * z + 2 * x * w;
        mat[2][2] = 1 - 2 * x * x - 2 * y * y;
    }

    public Matrix(float[] values, int rows, int cols)
    {
        if (rows*cols != values.Length)
        {
            throw new Exception("rows x cols: " + rows + "x" + cols + " = " + rows * cols + " does not equal given data of size " + values.Length);
        }

        mat = MatrixCreate(rows, cols);

        int pos = 0;
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                mat[i][j] = values[pos];
                pos++;
            }
        }
    }

    public Matrix(double[] values, int rows, int cols)
    {
        if (rows * cols != values.Length)
        {
            throw new Exception("rows x cols: " + rows + "x" + cols + " = " + rows * cols + " does not equal given data of size " + values.Length);
        }

        mat = MatrixCreate(rows, cols);

        int pos = 0;
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                mat[i][j] = (float)values[pos];
                pos++;
            }
        }
    }

    // From http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    // Uses only top left 3x3
    public Quaternion ToQuaterion()
    {
        float[][] a = mat;
        Quaternion q = new Quaternion();
        float trace = a[0][0] + a[1][1] + a[2][2]; // I removed + 1.0f; see discussion with Ethan
        if (trace > 0)
        {// I changed M_EPSILON to 0
            float s = 0.5f / Mathf.Sqrt(trace + 1.0f);
            q.w = 0.25f / s;
            q.x = (a[2][1] - a[1][2]) * s;
            q.y = (a[0][2] - a[2][0]) * s;
            q.z = (a[1][0] - a[0][1]) * s;
        }
        else
        {
            if (a[0][0] > a[1][1] && a[0][0] > a[2][2])
            {
                float s = 2.0f * Mathf.Sqrt(1.0f + a[0][0] - a[1][1] - a[2][2]);
                q.w = (a[2][1] - a[1][2]) / s;
                q.x = 0.25f * s;
                q.y = (a[0][1] + a[1][0]) / s;
                q.z = (a[0][2] + a[2][0]) / s;
            }
            else if (a[1][1] > a[2][2])
            {
                float s = 2.0f * Mathf.Sqrt(1.0f + a[1][1] - a[0][0] - a[2][2]);
                q.w = (a[0][2] - a[2][0]) / s;
                q.x = (a[0][1] + a[1][0]) / s;
                q.y = 0.25f * s;
                q.z = (a[1][2] + a[2][1]) / s;
            }
            else
            {
                float s = 2.0f * Mathf.Sqrt(1.0f + a[2][2] - a[0][0] - a[1][1]);
                q.w = (a[1][0] - a[0][1]) / s;
                q.x = (a[0][2] + a[2][0]) / s;
                q.y = (a[1][2] + a[2][1]) / s;
                q.z = 0.25f * s;
            }
        }
        return q;
    }

    public Vector4 ToVec4()
    {
        float[] arr = ToArray();

        if (!((rows == 4 && cols == 1) || (rows == 1 && cols == 4)))
        {
            throw new ArgumentException("Matrix is of dimension (" + rows + ", " + cols + ") which cannot be converted to a Vector4");
        }

        return new Vector4(arr[0], arr[1], arr[2], arr[3]);
    }

    public Vector3 ToVec3()
    {
        float[] arr = ToArray();

        if (!((rows == 3 && cols == 1) || (rows == 1 && cols == 3)))
        {
            throw new ArgumentException("Matrix is of dimension (" + rows + ", " + cols + ") which cannot be converted to a Vector3");
        }

        return new Vector3(arr[0], arr[1], arr[2]);
    }

    public Vector2 ToVec2()
    {
        float[] arr = ToArray();

        if (!((rows == 2 && cols == 1) || (rows == 1 && cols == 2)))
        {
            throw new ArgumentException("Matrix is of dimension (" + rows + ", " + cols + ") which cannot be converted to a Vector2");
        }

        return new Vector2(arr[0], arr[1]);
    }



    public override string ToString()
    {
        return MatrixAsString(mat);
    }

    public float[] ToArray()
    {
        float[] arr = new float[rows * cols];
        int pos = 0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                arr[pos++] = mat[i][j];
        return arr;
    }

    public float this[int i, int j]
    {
        get { return mat[i][j]; }
        set { mat[i][j] = value; }
    }

    public static Matrix operator *(Matrix m, float s)
    {
        Matrix res = new Matrix(m.rows, m.cols);
        for (int i = 0; i < m.rows; ++i)
            for (int j = 0; j < m.cols; ++j)
                res[i, j] = m[i, j] * s;
        return res;
    }

    public static Matrix operator *(float s, Matrix m)
    {
        return m * s;
    }

    public static Matrix operator +(Matrix m, float s)
    {
        Matrix res = new Matrix(m.rows, m.cols);
        for (int i = 0; i < m.rows; ++i)
            for (int j = 0; j < m.cols; ++j)
                res[i, j] = m[i, j] + s;
        return res;
    }

    public static Matrix operator +(float s, Matrix m)
    {
        return m + s;
    }

    public static Matrix operator -(Matrix m, float s)
    {
        return m + (-s);
    }

    public static Matrix operator -(float s, Matrix m)
    {
        return (-m) + s;
    }


    public static Matrix operator *(Matrix c1, Matrix c2)
    {
        return new Matrix(MatrixProduct(c1.mat, c2.mat));
    }

    public static Matrix operator *(Matrix c1, Vector3 c2)
    {
        return new Matrix(MatrixProduct(c1.mat, (new Matrix(c2)).mat));
    }
    public static Matrix operator *(Vector3 c1, Matrix c2)
    {
        return new Matrix(MatrixProduct((new Matrix(c1)).T.mat, c2.mat));
    }

    public static Matrix operator *(Matrix c1, Vector4 c2)
    {
        return new Matrix(MatrixProduct(c1.mat, (new Matrix(c2)).mat));
    }
    public static Matrix operator *(Vector4 c1, Matrix c2)
    {
        return new Matrix(MatrixProduct((new Matrix(c1)).T.mat, c2.mat));
    }

    public static Matrix operator -(Matrix m)
    {
        Matrix res = new Matrix(m.rows, m.cols);

        for (int i = 0; i < m.rows; ++i)
            for (int j = 0; j < m.cols; ++j)
                res[i, j] = -m[i, j];
        return res;
    }

    public static Matrix operator +(Matrix c1, Matrix c2)
    {
        if (c1.rows != c2.rows || c1.cols != c2.cols)
        {
            throw new ArgumentException("Left hand side size: (" + c1.rows + ", " + c1.cols + ") != Right hand side size: (" + c2.rows + ", " + c2.cols + ")");
        }
        Matrix res = new Matrix(c1.rows, c1.cols);
        for (int i = 0; i < c1.rows; ++i)
            for (int j = 0; j < c1.cols; ++j)
                res[i, j] = c1[i, j] + c2[i, j];
        return res;
    }

    public static Matrix operator -(Matrix c1, Matrix c2)
    {
        return c1 + (-c2);
    }

    public Matrix Inverse()
    {
        return new Matrix(MatrixInverse(mat));
    }


    public static float[][] MatrixInverse(float[][] matrix)
    {
        // assumes determinant is not 0
        // that is, the matrix does have an inverse
        int n = matrix.Length;
        float[][] result = MatrixCreate(n, n); // make a copy of matrix
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                result[i][j] = matrix[i][j];

        float[][] lum; // combined lower & upper
        int[] perm;
        int toggle;
        toggle = MatrixDecompose(matrix, out lum, out perm);

        float[] b = new float[n];
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
                if (i == perm[j])
                    b[j] = 1.0f;
                else
                    b[j] = 0.0f;

            float[] x = Helper(lum, b); // 
            for (int j = 0; j < n; ++j)
                result[j][i] = x[j];
        }
        return result;
    } // MatrixInverse

    public static int MatrixDecompose(float[][] m, out float[][] lum, out int[] perm)
    {
        // Crout's LU decomposition for matrix determinant and inverse
        // stores combined lower & upper in lum[][]
        // stores row permuations into perm[]
        // returns +1 or -1 according to even or odd number of row permutations
        // lower gets dummy 1.0s on diagonal (0.0s above)
        // upper gets lum values on diagonal (0.0s below)

        int toggle = +1; // even (+1) or odd (-1) row permutatuions
        int n = m.Length;

        // make a copy of m[][] into result lu[][]
        lum = MatrixCreate(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                lum[i][j] = m[i][j];


        // make perm[]
        perm = new int[n];
        for (int i = 0; i < n; ++i)
            perm[i] = i;

        for (int j = 0; j < n - 1; ++j) // process by column. note n-1 
        {
            float max = Math.Abs(lum[j][j]);
            int piv = j;

            for (int i = j + 1; i < n; ++i) // find pivot index
            {
                float xij = Math.Abs(lum[i][j]);
                if (xij > max)
                {
                    max = xij;
                    piv = i;
                }
            } // i

            if (piv != j)
            {
                float[] tmp = lum[piv]; // swap rows j, piv
                lum[piv] = lum[j];
                lum[j] = tmp;

                int t = perm[piv]; // swap perm elements
                perm[piv] = perm[j];
                perm[j] = t;

                toggle = -toggle;
            }

            float xjj = lum[j][j];
            if (xjj != 0.0)
            {
                for (int i = j + 1; i < n; ++i)
                {
                    float xij = lum[i][j] / xjj;
                    lum[i][j] = xij;
                    for (int k = j + 1; k < n; ++k)
                        lum[i][k] -= xij * lum[j][k];
                }
            }

        } // j

        return toggle;
    } // MatrixDecompose

    public static float[] Helper(float[][] luMatrix, float[] b) // helper
    {
        int n = luMatrix.Length;
        float[] x = new float[n];
        b.CopyTo(x, 0);

        for (int i = 1; i < n; ++i)
        {
            float sum = x[i];
            for (int j = 0; j < i; ++j)
                sum -= luMatrix[i][j] * x[j];
            x[i] = sum;
        }

        x[n - 1] /= luMatrix[n - 1][n - 1];
        for (int i = n - 2; i >= 0; --i)
        {
            float sum = x[i];
            for (int j = i + 1; j < n; ++j)
                sum -= luMatrix[i][j] * x[j];
            x[i] = sum / luMatrix[i][i];
        }

        return x;
    } // Helper

    public static float MatrixDeterminant(float[][] matrix)
    {
        float[][] lum;
        int[] perm;
        int toggle = MatrixDecompose(matrix, out lum, out perm);
        float result = toggle;
        for (int i = 0; i < lum.Length; ++i)
            result *= lum[i][i];
        return result;
    }

    // ----------------------------------------------------------------

    public static float[][] MatrixCreate(int rows, int cols)
    {
        float[][] result = new float[rows][];
        for (int i = 0; i < rows; ++i)
            result[i] = new float[cols];
        return result;
    }

    public static float[][] MatrixProduct(float[][] matrixA,
      float[][] matrixB)
    {
        int aRows = matrixA.Length;
        int aCols = matrixA[0].Length;
        int bRows = matrixB.Length;
        int bCols = matrixB[0].Length;
        if (aCols != bRows)
            throw new Exception("Non-conformable matrices");

        float[][] result = MatrixCreate(aRows, bCols);

        for (int i = 0; i < aRows; ++i) // each row of A
            for (int j = 0; j < bCols; ++j) // each col of B
                for (int k = 0; k < aCols; ++k) // could use k < bRows
                    result[i][j] += matrixA[i][k] * matrixB[k][j];

        return result;
    }

    public static string MatrixAsString(float[][] matrix)
    {
        string s = "";
        for (int i = 0; i < matrix.Length; ++i)
        {
            for (int j = 0; j < matrix[i].Length; ++j)
                s += matrix[i][j].ToString("F3").PadLeft(8) + " ";
            s += Environment.NewLine;
        }
        return s;
    }

    public static float[][] ExtractLower(float[][] lum)
    {
        // lower part of an LU Doolittle decomposition (dummy 1.0s on diagonal, 0.0s above)
        int n = lum.Length;
        float[][] result = MatrixCreate(n, n);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (i == j)
                    result[i][j] = 1.0f;
                else if (i > j)
                    result[i][j] = lum[i][j];
            }
        }
        return result;
    }

    public static float[][] ExtractUpper(float[][] lum)
    {
        // upper part of an LU (lu values on diagional and above, 0.0s below)
        int n = lum.Length;
        float[][] result = MatrixCreate(n, n);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (i <= j)
                    result[i][j] = lum[i][j];
            }
        }
        return result;
    }
} 
