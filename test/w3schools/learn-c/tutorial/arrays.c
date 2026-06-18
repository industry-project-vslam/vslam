#include <stdio.h>
#include <string.h> // for using string functions

int main(){
    // int myNumbers[] = {25, 50, 75, 100};
    // printf("%d", myNumbers[0]);

    // char greetings[] = "Hello World!";
    // printf("\n%s", greetings);
    
    // printf("\n%c", greetings[0]);

    // greetings[0] = 'J';
    // printf("\n%s\n", greetings);

    // char greetings_different[] = {'H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd', '!', '\0'};

    // printf("%zu\n", sizeof(greetings));   // Outputs 13
    // printf("%zu\n", sizeof(greetings_different));  // Outputs 13

    // char message[] = "Good to see you,";
    // char fname[] = "John";
    // printf("%s %s!", message, fname);

    // char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    // printf("%zu", strlen(alphabet));
    // printf("%zu\n", sizeof(alphabet)); // Note that sizeof and strlen behaves differently, as sizeof also includes the \0 character when counting:

    // char alphabet[50] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    // printf("%zu\n", strlen(alphabet));   // 26
    // printf("%zu\n", sizeof(alphabet));   // 50

    // char str1[20] = "Hello ";
    // char str2[] = "World!";
    // // Concatenate str2 to str1 (result is stored in str1)
    // strcat(str1, str2);
    // // Print str1
    // printf("%s", str1);

    // char str1[20] = "Hello World!";
    // char str2[20];
    // // Copy str1 to str2
    // strcpy(str2, str1);
    // // Print str2
    // printf("%s", str2);

    char str1[] = "Hello";
    char str2[] = "Hello";
    char str3[] = "Hi";
    // Compare str1 and str2, and print the result
    printf("%d\n", strcmp(str1, str2));  // Returns 0 (the strings are equal)
    // Compare str1 and str3, and print the result
    printf("%d\n", strcmp(str1, str3));  // Returns -4 (the strings are not equal)

    return 0;
}