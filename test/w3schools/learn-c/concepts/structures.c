#include <stdio.h>
#include <string.h>

// struct myStructure {
//   int myNum;
//   char myLetter;
//   char myString[30];
// };

// int main() {
//   // Create a structure variable and assign values to it
//   struct myStructure s1 = {13, 'B', "Some text"};
//   printf("%d %c %s\n", s1.myNum, s1.myLetter, s1.myString);

//   struct myStructure s2;
//   s2 = s1;
//   struct myStructure *ptr = &s1;
  
//   // Modify values
//   s1.myNum = 30;
//   s1.myLetter = 'C';
//   strcpy(s1.myString, "Something else");

//   // Print values
//   printf("%d %c %s\n", s1.myNum, s1.myLetter, s1.myString);
//   printf("%d %c %s\n", s2.myNum, s2.myLetter, s2.myString);
//   printf("%d %c %s\n", ptr->myNum, ptr->myLetter, ptr->myString);

//   return 0;
// }

/*struct Car {
  char brand[30];
  int year;
};

// Function that takes a pointer to a Car struct and updates the year
void updateYear(struct Car *c) {
  c->year = 2025;  // Change the year
}

int main() {
  struct Car myCar = {"Toyota", 2020};

  updateYear(&myCar);  // Pass a pointer so the function can change the year

  printf("Brand: %s\n", myCar.brand);
  printf("Year: %d\n", myCar.year);

  return 0;
}*/

/*Why Use Struct Pointers?

Using pointers with structs is helpful when:

    You want to avoid copying large amounts of data.
    Instead of copying a whole struct, you can just pass a pointer. This makes your program faster and uses less memory.

    You want to change values inside a function.
    If you pass a pointer to a struct into a function, the function can change the original values.

    You want to create structs dynamically using memory allocation.
    With pointers, you can use malloc() to create structs while the program is running.
    You will learn more about memory management in a later chapter.
*/

// union myUnion {
//   int myNum;
//   char myLetter;
//   char myString[37];
// };

// int main() {
//   union myUnion u1;

//   u1.myNum = 1000;

//   // Since this is the last value written to the union, myNum no longer holds 1000 - its value is now invalid
//   u1.myLetter = 'A';

//   printf("myNum: %d\n", u1.myNum); // This value is no longer reliable
//   printf("myLetter: %c\n", u1.myLetter); // Prints 'A'
//   printf("Size of union: %zu bytes\n", sizeof(u1));

//   return 0;
// }

/*Typedef*/


typedef float Temperature;

// Without typedef:
struct Car {
  char brand[30];
  int year;
};

// With typedef:
typedef struct {
  char brand[30];
  int year;
} Car;

int main() {
  Temperature today = 25.5;
  Temperature tomorrow = 18.6;

  printf("Today: %.1f C\n", today);
  printf("Tomorrow: %.1f C\n", tomorrow);

  struct Car car1 = {"BMW", 1999}; // needs "struct"
  Car car2 = {"Ford", 1969}; // shorter with typedef

  printf("%s %d\n", car1.brand, car1.year);
  printf("%s %d\n", car2.brand, car2.year);

  return 0;
}

// order struct mpembers high bytes to low bytes to reduce memory padding