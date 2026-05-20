// #include <stdio.h>

// void printHello(){
//     printf("Hello World!");
// }

// void calculateSum() {
//     int x = 5;
//     int y = 10;
//     int sum = x + y;
//     printf("The sum of x + y is: %d", sum);
// }

// void printName(char name[]) {
//     printf("Hello %s\n", name);
// }

// void printNameAge(char name[], int age) {
//     printf("Hello %s. You are %d years old.\n", name, age);
// }

// int main(){
//     printNameAge("Roel Remmerie", 22);
//     return 0;
// }

/*good practice = declaration -> main -> defenition

// Function declaration
void myFunction();

// The main method
int main() {
  myFunction();  // call the function
  return 0;
}

// Function definition
void myFunction() {
  printf("I just got executed!");
}

// Function declaration
int myFunction(int x, int y);

// The main method
int main() {
  int result = myFunction(5, 3); // call the function
  printf("Result is = %d", result);
  return 0;
}

// Function definition
int myFunction(int x, int y) {
  return x + y;
}

*/

/*Math*/

// #include <stdio.h>
// #include <math.h>

// int main(){
//     printf("%f\n", sqrt(16));
//     printf("%f\n", ceil(1.4));
//     printf("%f\n", floor(1.4));
//     printf("%f", pow(4, 3));
//     return 0;
// }

/*Inline Function

You might sometimes see the inline keyword used in other people's functions. It's not something you need to use often as a beginner, but it's good to know what it means.

An inline function is a small function that asks the compiler to insert its code directly where it is called, instead of jumping to it.

This can make short, frequently used functions a little faster, because it removes the small delay of a normal function call.

Let's compare a regular function with an inline function:*/


// // Regular Function
// int square(int x) {
//   return x * x;
// }

// // Inline Function
// inline int square(int x) {
//   return x * x;
// }

/*Regular Function 	Inline Function
Code jumps to the function each time it's called 	Code is inserted directly where it's called
Slightly slower (small delay) 	Slightly faster
Good for large functions 	Good for small functions*/

/*recursion*/

#include <stdio.h>

int sum(int k);

int main() {
  int result = sum(10);
  printf("%d", result);
  return 0;
}

int sum(int k) {
  if (k > 0) {
    return k + sum(k - 1); // recursion
  } else {
    return 0;
  }
}

/*The developer should be very careful with recursion
as it can be quite easy to slip into writing a function which never terminates,
or one that uses excess amounts of memory or processor power.
However, when written correctly, recursion can be a very efficient and mathematically-elegant approach to programming.*/