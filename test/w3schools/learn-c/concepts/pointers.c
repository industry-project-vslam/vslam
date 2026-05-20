/*function pointer*/

// #include <stdio.h>

// int add(int a, int b) {
//   return a + b;
// }

// int main() {
//   int (*ptr)(int, int) = add;
//   int result = ptr(5, 3);
//   printf("Result: %d", result);
//   return 0;
// }

/*callback*/ // used for event driven programs that need user-provided callback functions

// #include <stdio.h>

// void greetMorning() { printf("Good morning!\n"); }
// void greetEvening() { printf("Good evening!\n"); }

// void greet(void (*func)()) {
//   func();
// }

// int main() {
//   greet(greetMorning);
//   greet(greetEvening);
//   return 0;
// }

/*Function Pointer Array*/

/* You can also store several function pointers in an array, so you can choose which function to run while the program is running.

This example runs three different functions using an array of function pointers:
Example*/

// #include <stdio.h>

// void add() { printf("Add\n"); }
// void subtract() { printf("Subtract\n"); }
// void multiply() { printf("Multiply\n"); }

// int main() {
//   void (*operations[3])() = { add, subtract, multiply };
//   for (int i = 0; i < 3; i++) {
//     operations[i]();
//   }
//   return 0;
// }

/*This is often used for simple menus, command lists, or calculators - anywhere you want to call different functions based on user input.*/

// #include <stdio.h>

// void add(int a, int b) { printf("Result: %d\n", a + b); }
// void subtract(int a, int b) { printf("Result: %d\n", a - b); }
// void multiply(int a, int b) { printf("Result: %d\n", a * b); }

// int main() {
//   int choice, x = 10, y = 5;

//   void (*operations[3])(int, int) = { add, subtract, multiply };

//   printf("x = %d, y = %d\n\n", x, y);
//   printf("Choose an operation:\n");
//   printf("0: Add\n1: Subtract\n2: Multiply\n");
//   scanf("%d", &choice);

//   if (choice >= 0 && choice < 3) {
//     operations[choice](x, y);
//   } else {
//     printf("Invalid choice!\n");
//   }

//   return 0;
// }

/*Function Pointer vs Normal Function
Normal Function 	Function Pointer
Called directly by its name 	Called using a pointer
The function is decided before the program runs 	You can choose which function to call while the program is running
Good for simple code 	Good for flexible and reusable code*/

/*Summary

    A function pointer stores the address of a function.
    You can declare, assign, and call functions through it.
    It allows passing functions as arguments to other functions.
    Useful for callbacks, menus, and flexible program design.
*/

/*more callbacks*/

// #include <stdio.h>

// void sayHello() {
//   printf("Hello from the callback!\n");
// }

// void runCallback(void (*callback)()) {
//   printf("Before calling the callback...\n");
//   callback();
//   printf("After calling the callback.\n");
// }

// int main() {
//   runCallback(sayHello);
//   return 0;
// }

/*Callback with Parameters

You can also pass functions that take parameters - just make sure the function pointer type matches:
Example
Passing a function with parameters as a callback:*/

// #include <stdio.h>

// void addNumbers(int a, int b) {
//   printf("The sum is: %d\n", a + b);
// }

// void calculate(void (*callback)(int, int), int x, int y) {
//   callback(x, y);
// }

// int main() {
//   calculate(addNumbers, 5, 3);
//   return 0;
// }

/*Real-World Example: Using Callbacks in qsort()

Many C standard library functions use callbacks. For example, the qsort() function in <stdlib.h> uses a callback to compare elements while sorting.

You provide the comparison function, and qsort() calls it as needed. This will sort the elements:
Example*/

#include <stdio.h>
#include <stdlib.h>

int compare(const void *a, const void *b) {
  return (*(int*)a - *(int*)b);
}

int main() {
  int numbers[] = { 5, 2, 9, 1, 7 };
  int size = sizeof(numbers) / sizeof(numbers[0]);

  qsort(numbers, size, sizeof(int), compare);

  for (int i = 0; i < size; i++) {
    printf("%d ", numbers[i]);
  }
  return 0;
}

/*Here, compare() is the callback function used by qsort() to decide how to order the numbers.*/