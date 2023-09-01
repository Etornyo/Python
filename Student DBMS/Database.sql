DROP TABLE IF EXISTS users;

CREATE TABLE users(
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    Conact TEXT UNIQUE NOT NULL,
    adddress TEXT,
    email UNIQUE NOT NULL,
    date_of_birth DATE
);

-
CREATE TABLE pharmacy_admin IF NOT EXISTS(
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password TEXT  NOT NULL,
    email UNIQUE NOT NULL
);

CREATE TABLE inventories IF NOT EXISTS(
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    Quantity INT NOT NULL CHECK(Quantity >= 0),
    Price NUMERIC NOT NULL CHECK(Price > 0)

);



CREATE TABLE cart_item IF NOT EXISTS(
    user_id UUID REFERENCES users,
    inventory_id UUID REFERENCES inventory,
    Quantity INT NOT NULL CHECK(Quantity > 0),
    PRIMARY KEY (user_id,inventory_id)

);

CREATE TABLE checkout IF NOT EXISTS(
    id UUID PRIMARY KEY,
    cart_item_id UUID REFERENCES cart_item  

);


CREATE TABLE orders IF NOT EXISTS(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users,
    cart_item_id UUID REFERENCES cart_items,
    total_price NUMERIC CHECK(total_price > 0)
    status TEXT NOT NULL

);

