#include <stdio.h>

int main(){
    // // Create an integer variable that will store the number we get from the user
    // int myNum;
    // // Ask the user to type a number
    // printf("Type a number: \n");
    // // Get and save the number the user types
    // scanf("%d", &myNum);
    // // Output the number the user typed
    // printf("Your number is: %d", myNum);

    // // Create an int and a char variable
    // int myNum;
    // char myChar;

    // // Ask the user to type a number AND a character
    // printf("Type a number AND a character and press enter: \n");

    // // Get and save the number AND character the user types
    // scanf("%d %c", &myNum, &myChar);

    // // Print the number
    // printf("Your number is: %d\n", myNum);

    // // Print the character
    // printf("Your character is: %c\n", myChar);

    // // Create a string
    // char fullName[30];
    // // Ask the user to input some text
    // printf("Enter your full name: \n");
    // // Get and save the text
    // scanf("%s", fullName); //However, the scanf() function has some limitations: it considers space (whitespace, tabs, etc) as a terminating character, which means that it can only display a single word (even if you type many words).
    // // Output the text
    // printf("Hello %s", fullName);

    /*From the example above, you would expect the program to print "John Doe", but it only prints "John".
    That's why, when working with strings, we often use the fgets() function to read a line of text.
    Note that you must include the following arguments: the name of the string variable, sizeof(string_name), and stdin:*/

    char fullName[30];

    printf("Type your full name: \n");
    fgets(fullName, sizeof(fullName), stdin);

    printf("Hello %s", fullName);

    // Type your full name: John Doe
    // Hello John Doe

    return 0;
}